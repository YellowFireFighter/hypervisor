//! The model-specific registers the host- and guest-state programming reads,
//! and a thin read over them.
//!
//! Nothing here decides anything; it is the one place the register numbers the
//! rest of the crate needs are named, so a bare `0xC000_0080` never appears at
//! a use site.

use x86_64::registers::model_specific::Msr;

/// `IA32_EFER`, the extended feature register the guest and host run with.
pub(crate) const IA32_EFER: u32 = 0xC000_0080;
/// `IA32_FS_BASE`, the base the host's `FS` is addressed from.
pub(crate) const IA32_FS_BASE: u32 = 0xC000_0100;
/// `IA32_GS_BASE`, the base the host's `GS` is addressed from.
pub(crate) const IA32_GS_BASE: u32 = 0xC000_0101;

/// Reads a model-specific register.
///
/// # Safety
///
/// `number` must be a register this processor implements.
pub(crate) unsafe fn rdmsr(number: u32) -> u64 {
    // SAFETY: the caller guarantees the register exists; a read has no other
    // precondition.
    unsafe { Msr::new(number).read() }
}
