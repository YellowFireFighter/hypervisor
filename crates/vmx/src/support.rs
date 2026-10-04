//! Whether this machine can enter VMX operation at all.
//!
//! Two gates stand between a processor and `VMXON`, and they answer different
//! questions. `CPUID` says whether the silicon *has* the extension;
//! [`IA32_FEATURE_CONTROL`] says whether firmware *left it usable*, because a
//! locked feature-control register can forbid `VMXON` on a processor that
//! otherwise supports it. Telling the two apart is the difference between "this
//! machine cannot" and "this machine was configured not to" — the same
//! distinction `svm`'s `VM_CR` draws for AMD, and the same two things a
//! hypervisor owes an accurate word about to whoever is trying to boot it.
//!
//! Nothing here executes `CPUID` or reads the register: a caller brings the
//! `ECX` result and the register value, and this decodes them.

use bitflags::bitflags;

/// Leaf 1 of `CPUID`, whose `ECX` carries the VMX feature bit.
pub const FEATURE_LEAF: u32 = 1;

/// `CPUID.1:ECX[5]`, set when the processor supports VMX.
const VMX_FEATURE_BIT: u32 = 1 << 5;

/// Whether `CPUID.1:ECX` reports VMX support.
///
/// The whole of what the silicon has to say; whether VMX may actually be
/// turned on is then [`FeatureControl`]'s question.
#[must_use]
pub const fn supported(cpuid_leaf1_ecx: u32) -> bool {
    cpuid_leaf1_ecx & VMX_FEATURE_BIT != 0
}

/// The number of `IA32_FEATURE_CONTROL`.
pub const IA32_FEATURE_CONTROL: u32 = 0x3A;

bitflags! {
    /// `IA32_FEATURE_CONTROL`, the register firmware locks VMX behind.
    ///
    /// While [`LOCK`](Self::LOCK) is clear the register is writable and
    /// `VMXON` faults regardless of the other bits, so firmware sets the bits
    /// it permits and then sets the lock; software that finds it unlocked may
    /// configure it itself. Only the three bits that bear on VMX are modelled;
    /// the rest govern unrelated features and are preserved by
    /// [`from_bits_retain`](Self::from_bits_retain).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FeatureControl: u64 {
        /// Locks the register. Until it is set the register is writable and
        /// `VMXON` is not permitted.
        const LOCK = 1 << 0;
        /// Permits `VMXON` in SMX operation (inside a measured launch).
        const VMXON_IN_SMX = 1 << 1;
        /// Permits `VMXON` outside SMX operation, which is where a type-1
        /// hypervisor runs.
        const VMXON_OUTSIDE_SMX = 1 << 2;
    }
}

impl FeatureControl {
    /// Whether this register, as firmware left it, permits `VMXON` outside SMX.
    ///
    /// Both the lock and the outside-SMX permission must be set: an unlocked
    /// register forbids `VMXON` whatever else it says, and the permission bit
    /// is what the lock then makes binding.
    #[must_use]
    pub const fn vmxon_allowed_outside_smx(self) -> bool {
        self.contains(Self::LOCK.union(Self::VMXON_OUTSIDE_SMX))
    }
}

#[cfg(test)]
mod tests {
    use super::{FeatureControl, IA32_FEATURE_CONTROL, supported};

    #[test]
    fn the_register_number_is_the_architectural_one() {
        assert_eq!(IA32_FEATURE_CONTROL, 0x3A);
    }

    #[test]
    fn vmx_support_is_bit_five_of_ecx() {
        assert!(supported(1 << 5));
        assert!(!supported(!(1 << 5)));
    }

    #[test]
    fn vmxon_needs_the_lock_and_the_outside_smx_bit() {
        let both = FeatureControl::LOCK | FeatureControl::VMXON_OUTSIDE_SMX;
        assert!(both.vmxon_allowed_outside_smx());
        // Permission without the lock does not count: an unlocked register
        // forbids VMXON whatever its other bits say.
        assert!(!FeatureControl::VMXON_OUTSIDE_SMX.vmxon_allowed_outside_smx());
        // Locked to SMX-only is a configured refusal.
        assert!(!(FeatureControl::LOCK | FeatureControl::VMXON_IN_SMX).vmxon_allowed_outside_smx());
    }

    #[test]
    fn unmodelled_bits_are_preserved() {
        let raw = 0xDEAD_0000 | FeatureControl::LOCK.bits();
        let flags = FeatureControl::from_bits_retain(raw);
        assert_eq!(flags.bits(), raw);
        assert!(flags.contains(FeatureControl::LOCK));
    }
}
