//! The VMX instructions, wrapped so their flag convention becomes an
//! [`Outcome`].
//!
//! Each wrapper executes one instruction and captures `RFLAGS` immediately
//! after it, before anything else can disturb the flags, and turns that into an
//! [`Outcome`]. Nothing here is reachable under test — these instructions fault
//! outside VMX operation — so this module is compiled and reviewed but not
//! run on the host; the flag decoding it depends on is tested in
//! [`crate::error`].
//!
//! # Why the physical-address operands are taken by value and spilled
//!
//! `VMXON`, `VMCLEAR` and `VMPTRLD` do not take the physical address in a
//! register: they take a memory operand that *holds* the address. So each
//! wrapper writes the address into a local and hands the instruction a pointer
//! to it, which is why those three touch the stack and are not `nomem`.

use vmx::FieldEncoding;
use x86_64::PhysAddr;

use crate::error::{Outcome, VmFail};

/// Enters VMX operation with the VMXON region at `region`.
///
/// # Safety
///
/// The processor must support VMX and have `CR4.VMXE` set and
/// `IA32_FEATURE_CONTROL` permitting `VMXON`; `region` must be the physical
/// address of a page whose first doubleword holds this processor's VMCS
/// revision identifier and which nothing else uses while VMX operation lasts.
/// The caller owns leaving VMX operation with [`vmxoff`].
#[must_use]
pub unsafe fn vmxon(region: PhysAddr) -> Outcome {
    let address = region.as_u64();
    let flags: u64;
    // SAFETY: the caller guarantees VMX is permitted and `region` names a
    // valid, exclusively-owned VMXON region; `address` is a live local for the
    // duration of the instruction, which reads the 8-byte operand from it.
    unsafe {
        core::arch::asm!(
            "vmxon [{address}]",
            "pushfq",
            "pop {flags}",
            address = in(reg) core::ptr::addr_of!(address),
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

/// Leaves VMX operation.
///
/// # Safety
///
/// The processor must be in VMX operation, entered by a prior [`vmxon`] on this
/// processor, and no VMCS may be current (clear it with [`vmclear`] first).
#[must_use]
pub unsafe fn vmxoff() -> Outcome {
    let flags: u64;
    // SAFETY: the caller guarantees this processor is in VMX operation with no
    // current VMCS.
    unsafe {
        core::arch::asm!(
            "vmxoff",
            "pushfq",
            "pop {flags}",
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

/// Clears the VMCS at `vmcs`, making it inactive and not current and flushing
/// any of its state the processor had cached.
///
/// # Safety
///
/// This processor must be in VMX operation and `vmcs` must be the physical
/// address of a page sized and aligned as a VMCS whose first doubleword holds
/// this processor's revision identifier.
#[must_use]
pub unsafe fn vmclear(vmcs: PhysAddr) -> Outcome {
    let address = vmcs.as_u64();
    let flags: u64;
    // SAFETY: the caller guarantees VMX operation and that `vmcs` names a
    // valid VMCS region; `address` is a live local the instruction reads its
    // operand from.
    unsafe {
        core::arch::asm!(
            "vmclear [{address}]",
            "pushfq",
            "pop {flags}",
            address = in(reg) core::ptr::addr_of!(address),
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

/// Makes the VMCS at `vmcs` the current one, the target of every later
/// [`vmread`] and [`vmwrite`] and of the next guest entry.
///
/// # Safety
///
/// This processor must be in VMX operation and `vmcs` must have been cleared
/// with [`vmclear`] since it was allocated, with a matching revision
/// identifier in its first doubleword.
#[must_use]
pub unsafe fn vmptrld(vmcs: PhysAddr) -> Outcome {
    let address = vmcs.as_u64();
    let flags: u64;
    // SAFETY: the caller guarantees VMX operation and that `vmcs` names a
    // cleared, revision-matched VMCS region; `address` is a live local.
    unsafe {
        core::arch::asm!(
            "vmptrld [{address}]",
            "pushfq",
            "pop {flags}",
            address = in(reg) core::ptr::addr_of!(address),
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

/// Reads the current VMCS's `field`.
///
/// # Errors
///
/// [`VmFail::Invalid`] if there is no current VMCS, or [`VmFail::Valid`] if the
/// field encoding is not one this processor's VMCS has.
///
/// # Safety
///
/// This processor must be in VMX operation with a VMCS made current by
/// [`vmptrld`].
pub unsafe fn vmread(field: FieldEncoding) -> Result<u64, VmFail> {
    let value: u64;
    let flags: u64;
    // SAFETY: the caller guarantees VMX operation with a current VMCS; the
    // instruction only reads processor state into `value`.
    unsafe {
        core::arch::asm!(
            "vmread {value}, {field}",
            "pushfq",
            "pop {flags}",
            value = out(reg) value,
            field = in(reg) u64::from(field.bits()),
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags).ok().map(|()| value)
}

/// Writes `value` into the current VMCS's `field`.
///
/// # Errors
///
/// [`VmFail::Invalid`] if there is no current VMCS, or [`VmFail::Valid`] if the
/// field is read-only or not one this processor's VMCS has.
///
/// # Safety
///
/// This processor must be in VMX operation with a VMCS made current by
/// [`vmptrld`].
pub unsafe fn vmwrite(field: FieldEncoding, value: u64) -> Result<(), VmFail> {
    let flags: u64;
    // SAFETY: the caller guarantees VMX operation with a current VMCS; the
    // instruction writes only to that structure.
    unsafe {
        core::arch::asm!(
            "vmwrite {field}, {value}",
            "pushfq",
            "pop {flags}",
            field = in(reg) u64::from(field.bits()),
            value = in(reg) value,
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags).ok()
}

/// Launches the guest the current VMCS describes, for the first time since it
/// was made current.
///
/// On success control passes to the guest and does not return here; the
/// processor continues at the VMCS's host `RIP` when the guest exits. This
/// wrapper returns only when the launch *failed* without entering the guest,
/// which is why it returns an [`Outcome`] rather than diverging.
///
/// # Safety
///
/// This processor must be in VMX operation with a fully initialized current
/// VMCS whose host `RIP` and host `RSP` name a valid return context, and whose
/// launch state is clear (a freshly [`vmclear`]ed VMCS). It does not preserve
/// or restore the guest's general registers — a complete run loop loads them
/// before this and saves them at the host `RIP` it returns through.
#[must_use]
pub unsafe fn vmlaunch() -> Outcome {
    let flags: u64;
    // SAFETY: the caller guarantees a fully initialized current VMCS and owns
    // the register discipline around entry; on failure the instruction only
    // sets flags.
    unsafe {
        core::arch::asm!(
            "vmlaunch",
            "pushfq",
            "pop {flags}",
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

/// Resumes the guest the current VMCS describes, which must already have been
/// launched.
///
/// Returns only on failure, as [`vmlaunch`] does.
///
/// # Safety
///
/// As [`vmlaunch`], except the current VMCS must be in the launched state —
/// this is the entry used for every guest entry after the first.
#[must_use]
pub unsafe fn vmresume() -> Outcome {
    let flags: u64;
    // SAFETY: the caller guarantees a launched current VMCS and the register
    // discipline around entry; on failure the instruction only sets flags.
    unsafe {
        core::arch::asm!(
            "vmresume",
            "pushfq",
            "pop {flags}",
            flags = out(reg) flags,
        );
    }
    Outcome::from_flags(flags)
}

#[cfg(test)]
mod tests {
    //! These wrappers cannot execute off a VMX-capable processor in VMX
    //! operation, so there is nothing to run here. That the module compiles —
    //! the operand orders, the flag capture, the field-encoding widening — is
    //! what this crate's build checks; the flag decoding the wrappers feed is
    //! tested in [`crate::error`].
}
