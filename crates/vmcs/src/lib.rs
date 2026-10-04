//! Entering VMX operation and driving a VMCS, on one processor.
//!
//! This is the Intel counterpart to the executable half of `vcpu`: where `vcpu`
//! turns AMD's extension on and runs a guest through `VMRUN`, this turns
//! Intel's on and talks to a VMCS through `VMXON`, `VMPTRLD`,
//! `VMREAD`/`VMWRITE` and the entry instructions. It is built on the `vmx`
//! definitions the way `vcpu` is built on `svm`.
//!
//! # Unverified
//!
//! Every instruction this crate issues faults outside VMX operation on a
//! VMX-capable processor, which no test host here is. So unlike the `vmx`
//! definitions, the instruction-issuing code in [`instr`] and [`enable`] is
//! compiled and reviewed but **not executed** anywhere yet — it has never run
//! on real hardware. What *is* tested, on the host, are the pure decisions it
//! is built from: the flag decoding in [`error`] and the control-register
//! reconciliation in [`fixed`]. Treat the rest as unproven until it has entered
//! VMX operation on an Intel processor.
//!
//! # Shape
//!
//! - [`error`] turns a VMX instruction's flag result into an [`Outcome`], and
//!   names the ways entering VMX operation can fail.
//! - [`fixed`] reconciles a control register against the bits VMX forces.
//! - [`instr`] is the VMX instructions, each wrapped to report an [`Outcome`].
//! - [`enable`] puts a processor into VMX operation, tying those together.

#![no_std]

pub mod enable;
pub mod error;
pub mod fixed;
pub mod instr;

pub use crate::{
    enable::{Vmx, enter},
    error::{EnterError, Outcome, VmFail},
};
