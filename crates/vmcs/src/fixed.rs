//! The control-register bits VMX forces on, and the reconciliation that
//! applies them.
//!
//! Before `VMXON`, and for the guest's own control registers, `CR0` and `CR4`
//! must have certain bits set and certain bits clear — the processor fixes
//! which through four read-only registers, `IA32_VMX_CR0_FIXED0`/`FIXED1` and
//! the `CR4` pair. A bit set in the `FIXED0` register must be set; a bit clear
//! in the `FIXED1` register must be clear; the rest are free. A `VMXON` or a VM
//! entry with a control register outside those bounds faults, so a value is
//! always reconciled against the pair first.
//!
//! This is the same shape as the control-word reconciliation in
//! [`vmx::Capability`], and like it, it is pure and tested on the host.

/// The number of `IA32_VMX_CR0_FIXED0`: bits that must be set in `CR0`.
pub const IA32_VMX_CR0_FIXED0: u32 = 0x486;
/// The number of `IA32_VMX_CR0_FIXED1`: bits that may be set in `CR0`.
pub const IA32_VMX_CR0_FIXED1: u32 = 0x487;
/// The number of `IA32_VMX_CR4_FIXED0`: bits that must be set in `CR4`.
pub const IA32_VMX_CR4_FIXED0: u32 = 0x488;
/// The number of `IA32_VMX_CR4_FIXED1`: bits that may be set in `CR4`.
pub const IA32_VMX_CR4_FIXED1: u32 = 0x489;

/// The control-register value `value` reconciled against a fixed-bit pair:
/// every bit the `fixed0` register forces on is set, every bit the `fixed1`
/// register forbids is cleared, and the rest are left as `value` had them.
///
/// `fixed0` and `fixed1` are the raw readings of the two registers for the
/// register being reconciled. The reconciliation is the same expression VMX
/// control words use — force the ones, mask to the ones that are allowed —
/// because the two registers describe exactly that: a lower bound and an upper
/// bound on the bits.
#[must_use]
pub const fn reconcile(value: u64, fixed0: u64, fixed1: u64) -> u64 {
    (value | fixed0) & fixed1
}

#[cfg(test)]
mod tests {
    use super::reconcile;

    #[test]
    fn forced_bits_are_set_and_forbidden_bits_cleared() {
        // Bit 0 must be set (fixed0 bit 0), bit 5 must be clear (fixed1 bit 5
        // is 0). Everything else is free.
        let fixed0 = 0b0000_0001;
        let fixed1 = !0b0010_0000_u64;

        // A value with bit 0 clear and bit 5 set gets both corrected.
        assert_eq!(reconcile(0b0010_0000, fixed0, fixed1), 0b0000_0001);
        // A value already within bounds is unchanged.
        assert_eq!(reconcile(0b0000_1001, fixed0, fixed1), 0b0000_1001);
    }

    #[test]
    fn protected_mode_and_paging_are_the_classic_cr0_forced_bits() {
        // On every VMX processor CR0.PE (bit 0) and CR0.PG (bit 31) are forced
        // on outside unrestricted guest, so a CR0 of zero comes back with them.
        let fixed0 = (1 << 0) | (1 << 31);
        let fixed1 = !0;
        assert_eq!(reconcile(0, fixed0, fixed1), (1 << 0) | (1 << 31));
    }
}
