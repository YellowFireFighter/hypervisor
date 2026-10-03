//! The structures and encodings Intel's VMX extension is programmed through,
//! stated once and checked where the compiler can check them.
//!
//! This is the Intel counterpart to the `svm` crate, and the two are kept
//! deliberately parallel: a hypervisor that runs on either machine wants the
//! same things of both — a description of a guest, a reason it stopped, the
//! switches that turn the extension on — and the only honest way to share the
//! code above them is to state each architecture's own structures in the same
//! shape underneath. Where `svm` describes AMD's secure virtual machine, this
//! describes Intel's virtual-machine extensions.
//!
//! # The one structural difference from `svm`
//!
//! AMD describes a guest with a control block a hypervisor reads and writes by
//! field offset: the layout is the ABI. Intel does not. The control structure —
//! the VMCS — is opaque: its format is the processor's own, it is never read by
//! offset, and the only way to touch a field is the `VMREAD` and `VMWRITE`
//! instructions naming that field by a 32-bit *encoding*. So the heart of this
//! crate is not a set of `#[repr(C)]` layouts but [`field`]: the encoding
//! scheme and a name for every field a hypervisor programs. Getting a VMCS
//! field encoding wrong fails exactly the way a wrong VMCB offset does — it
//! writes a plausible value into the wrong place and faults much later — which
//! is why the encodings are stated here once and tested rather than written out
//! at each `VMREAD`/`VMWRITE` site.
//!
//! # Definitions only
//!
//! Like `svm`, this crate holds no `unsafe`, touches no hardware, and executes
//! no instruction. It decodes values a caller brings — a `CPUID` result, a
//! model-specific register, a field encoding — and says what they mean.
//! Executing `VMXON`, allocating a VMCS, running a guest with `VMLAUNCH` and
//! reading back why it exited are all somebody else's work, built on top of
//! these definitions.
//!
//! # Shape
//!
//! - [`field`] is the VMCS field encoding scheme and the fields themselves.
//! - [`basic`] is `IA32_VMX_BASIC`, the register describing this processor's
//!   VMCS: its revision, its size, and the memory type it is accessed with.
//! - [`support`] decodes whether this machine can enter VMX operation at all,
//!   from the `CPUID` feature bit and the feature-control register firmware
//!   locks it behind.
//! - [`control`] is what the processor does on the guest's behalf and what it
//!   intercepts, together with the reconciliation against the capability
//!   registers that every control word must pass.
//! - [`exit`] is why a guest stopped, and [`event`] is what is handed to it on
//!   the way back in.

#![no_std]

pub mod basic;
pub mod control;
pub mod event;
pub mod exit;
pub mod field;
pub mod support;

pub use crate::{
    basic::VmxBasic,
    control::{Capability, PinBased, PrimaryProc, SecondaryProc, VmEntry, VmExit},
    event::Interruption,
    exit::{BasicExitReason, ExitReason},
    field::{Access, Field, FieldEncoding, Kind, Width},
    support::FeatureControl,
};

/// Bytes in the page the VMXON region and every VMCS are aligned to and sized
/// within.
///
/// Both regions are given to the processor as physical addresses with the low
/// twelve bits ignored, and `IA32_VMX_BASIC` never reports a VMCS larger than
/// one of these — so a region is always exactly a page, and anything less than
/// page alignment is a different address rather than a smaller mistake.
pub const PAGE_BYTES: usize = 4096;
