//! What a guest's VMX exits mean.
//!
//! This is the Intel counterpart to the `exits` crate, and it has the same job:
//! `vmcs`'s [`run`](vmcs::run) runs a guest and hands back every exit without
//! interpreting one, because what an exit means is policy. This is that policy
//! — the decision, for each exit, of whether the guest carries on and what the
//! host must do first.
//!
//! # What is answered, and how
//!
//! - `CPUID`, run on the machine and its result placed in the guest's
//!   registers. This layer forwards the processor's own answer unchanged; the
//!   concealment the AMD side applies — hiding the virtualization extension and
//!   the hypervisor leaves — is a policy edit that belongs on top of this
//!   dispatch, not in the act of taking the exit.
//! - `RDMSR` and `WRMSR`, forwarded to the machine's own register through
//!   [`probe`], so a guest that names a register the machine does not have is
//!   refused rather than faulting the host. A refusal stops the guest here,
//!   because giving it the general-protection fault it is owed needs the event
//!   injection a later layer builds.
//! - `VMCALL`, the guest's deliberate call into the host, handed back to the
//!   caller to answer — the VMX `VMMCALL`.
//! - `HLT` and a triple fault, each of which stops the guest: a halted guest
//!   has nothing to resume into, and a triple fault is unrecoverable.
//! - An EPT violation and a control-register access, decoded from the exit
//!   qualification and reported. Neither has a handler yet — a guest's own
//!   memory and control-register virtualization are later layers — so each
//!   stops the guest with the decoded reason rather than resuming blindly.
//!
//! Anything else stops the guest too. A hypervisor that resumed an exit it did
//! not understand would resume a guest whose state it had not fixed, and the
//! same exit would arrive again forever.
//!
//! # What has run, and what has not
//!
//! The pure decoding here — the control-register and EPT-violation exit
//! qualifications — is tested on the host. The dispatch itself drives the VMX
//! instructions through [`vmcs`] and so runs only in VMX operation on an Intel
//! processor; it has emulated a guest's `CPUID` and `RDMSR` exits there, each
//! driven through [`dispatch`] and checked against the machine's own answer.
//! `WRMSR` forwarding, and the EPT-violation and control-register reporting
//! paths, are written but have not been reached by a guest that exercises them.

#![no_std]

pub mod control_register;
mod cpuid;
mod msr;
pub mod violation;

use vmcs::{Registers, VmFail, Vmcs};
use vmx::{BasicExitReason, ExitReason, Field};

/// What the host decided to do about a guest's exit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Enter the guest again; the exit was answered.
    Resume,
    /// The guest made a `VMCALL`; what it means is the caller's to answer.
    Vmcall,
    /// The guest cannot be resumed; the reason says why.
    Stop(Stop),
}

/// Why a guest stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The guest executed `HLT` and has nothing to be resumed into.
    Halted,
    /// A fault while delivering a fault that itself faulted: unrecoverable.
    TripleFault,
    /// VM entry was rejected without running the guest; the basic reason names
    /// which consistency check the rejected state failed.
    EntryFailure(BasicExitReason),
    /// The guest accessed memory the EPT entries forbade. No handler maps a
    /// guest's own memory yet, so the decoded access is reported and the guest
    /// stops.
    EptViolation(violation::Violation),
    /// The guest accessed a control register. No control-register
    /// virtualization exists yet, so the decoded access is reported and the
    /// guest stops.
    ControlRegister(control_register::Access),
    /// The guest named a model-specific register the machine does not have.
    /// Delivering the general-protection fault it is owed needs event
    /// injection, which a later layer builds.
    MsrUnavailable(u32),
    /// A VMX instruction the dispatch itself issued was refused — reading the
    /// exit reason or advancing past the instruction failed.
    Vmcs(VmFail),
    /// An exit this layer does not yet answer.
    Unhandled(BasicExitReason),
}

/// Answers the exit the current VMCS records, editing `registers` as the exit
/// requires and returning whether the guest may be resumed.
///
/// On [`Flow::Resume`] the guest's registers carry whatever the exit produced
/// and its `RIP` has been advanced past the instruction that exited, so the
/// caller re-enters without further work. [`Flow::Vmcall`] leaves the `RIP` on
/// the `VMCALL`, for the caller to advance once it has answered.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation, and
/// `registers` must be the block [`vmcs::run::run`] filled on the exit being
/// answered. [`probe::install`] must have claimed the general-protection
/// vector, which the `RDMSR`/`WRMSR` forwarding relies on.
#[must_use]
pub unsafe fn dispatch(cell: &Vmcs, registers: &mut Registers) -> Flow {
    // SAFETY: the caller guarantees the current VMCS on this processor.
    let reason = match unsafe { cell.read(Field::EXIT_REASON) } {
        Ok(bits) => ExitReason::from_bits((bits & 0xFFFF_FFFF) as u32),
        Err(fail) => return Flow::Stop(Stop::Vmcs(fail)),
    };
    if reason.entry_failed() {
        return Flow::Stop(Stop::EntryFailure(reason.basic()));
    }

    match reason.basic() {
        BasicExitReason::CPUID => {
            cpuid::emulate(registers);
            // SAFETY: the exit was on CPUID, whose length the processor
            // recorded, and `cell` is current.
            unsafe { advance(cell) }
        }
        BasicExitReason::RDMSR => match msr::read(registers) {
            // SAFETY: the exit was on RDMSR, whose length the processor
            // recorded, and `cell` is current.
            Ok(()) => unsafe { advance(cell) },
            Err(msr) => Flow::Stop(Stop::MsrUnavailable(msr)),
        },
        BasicExitReason::WRMSR => match msr::write(registers) {
            // SAFETY: the exit was on WRMSR, whose length the processor
            // recorded, and `cell` is current.
            Ok(()) => unsafe { advance(cell) },
            Err(msr) => Flow::Stop(Stop::MsrUnavailable(msr)),
        },
        BasicExitReason::VMCALL => Flow::Vmcall,
        BasicExitReason::HLT => Flow::Stop(Stop::Halted),
        BasicExitReason::TRIPLE_FAULT => Flow::Stop(Stop::TripleFault),
        BasicExitReason::EPT_VIOLATION => {
            // SAFETY: `cell` is current.
            match unsafe { cell.read(Field::EXIT_QUALIFICATION) } {
                Ok(bits) => Flow::Stop(Stop::EptViolation(violation::Violation::decode(bits))),
                Err(fail) => Flow::Stop(Stop::Vmcs(fail)),
            }
        }
        BasicExitReason::CONTROL_REGISTER_ACCESS => {
            // SAFETY: `cell` is current.
            match unsafe { cell.read(Field::EXIT_QUALIFICATION) } {
                Ok(bits) => Flow::Stop(Stop::ControlRegister(control_register::Access::decode(
                    bits,
                ))),
                Err(fail) => Flow::Stop(Stop::Vmcs(fail)),
            }
        }
        other => Flow::Stop(Stop::Unhandled(other)),
    }
}

/// Advances the guest past the instruction it exited on, turning the result
/// into a [`Flow`].
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, and the last exit must
/// have been on an instruction whose length the processor recorded.
unsafe fn advance(cell: &Vmcs) -> Flow {
    // SAFETY: the caller guarantees the current VMCS and an instruction exit.
    match unsafe { cell.advance_past_instruction() } {
        Ok(()) => Flow::Resume,
        Err(fail) => Flow::Stop(Stop::Vmcs(fail)),
    }
}
