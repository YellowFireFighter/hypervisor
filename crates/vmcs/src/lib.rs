//! Entering VMX operation and driving a VMCS, on one processor.
//!
//! This is the Intel counterpart to the executable half of `vcpu`: where `vcpu`
//! turns AMD's extension on and runs a guest through `VMRUN`, this turns
//! Intel's on and talks to a VMCS through `VMXON`, `VMPTRLD`,
//! `VMREAD`/`VMWRITE` and the entry instructions. It is built on the `vmx`
//! definitions the way `vcpu` is built on `svm`.
//!
//! # What has run, and what has not
//!
//! The enabling path and the core instruction wrappers have been exercised in
//! VMX operation on an Intel processor: [`enter`] takes a processor into VMX
//! operation — the capability checks, the control-register reconciliation,
//! `CR4.VMXE` and `VMXON` — and [`Vmcs::activate`] makes a VMCS current through
//! `VMCLEAR` and `VMPTRLD`, after which a [`Vmcs::write`] and [`Vmcs::read`]
//! round-trip a field, and `VMXOFF` leaves cleanly. Those instructions —
//! `VMXON`, `VMCLEAR`, `VMPTRLD`, `VMWRITE`, `VMREAD`, `VMXOFF` — and the
//! reconciliation and flag decoding they rest on are therefore proven, not just
//! compiled.
//!
//! What has *not* run is guest entry: `VMLAUNCH` and `VMRESUME` are wrapped in
//! [`instr`] but no run loop drives them yet, so they are still unproven. The
//! pure decisions throughout — the flag decoding in [`error`] and the
//! control-register reconciliation in [`fixed`] — are also tested on the host.
//!
//! # Shape
//!
//! - [`error`] turns a VMX instruction's flag result into an [`Outcome`], and
//!   names the ways entering VMX operation can fail.
//! - [`fixed`] reconciles a control register against the bits VMX forces.
//! - [`instr`] is the VMX instructions, each wrapped to report an [`Outcome`].
//! - [`enable`] puts a processor into VMX operation, tying those together.
//! - [`vmcs`] is a VMCS made current, with typed field access and the
//!   capability reconciliation a control word is written through.
//!
//! # What is not here yet
//!
//! The guest run loop — the `VMLAUNCH`/`VMRESUME` sequence that saves and
//! restores the guest's general registers around entry — is deliberately not
//! written. It is register-allocation-constrained inline assembly whose
//! correctness cannot be established without running it in VMX operation on an
//! Intel processor, which nothing in this environment can do, so fabricating it
//! would be guesswork presented as a run loop. It is named here as the next
//! piece rather than left as a stub, together with the host- and guest-state
//! programming that fills a VMCS before that loop, the EPT mapper (the `npt`
//! counterpart) and VMX exit dispatch (the `exits` counterpart).

#![no_std]

pub mod enable;
pub mod error;
pub mod fixed;
pub mod instr;
pub mod vmcs;

pub use crate::{
    enable::{Vmx, enter},
    error::{EnterError, Outcome, VmFail},
    vmcs::Vmcs,
};
