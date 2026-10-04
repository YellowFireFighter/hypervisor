//! The access rights a guest segment is described by in the VMCS.
//!
//! Every guest segment appears in the VMCS as a selector, a base, a limit and
//! an access-rights doubleword, and the access-rights doubleword is the awkward
//! one. It carries the same fields a descriptor's attribute byte does — the
//! type, the descriptor and present bits, the privilege level, the long and
//! default-size and granularity bits — but laid out differently from both the
//! descriptor and AMD's packed attributes: the fields sit with a reserved gap
//! above the present bit, and there is an extra bit, *unusable*, that the
//! descriptor has no room for.
//!
//! That unusable bit is the reason this needs stating carefully. VMX has no
//! null selector the way a descriptor table does; a segment the guest is not
//! using is marked by setting this bit, and a VM entry checks the other fields
//! only for a segment that is not unusable. Getting it wrong fails entry rather
//! than faulting later, so the layout is modelled exactly and the bit is named.

use bitfield_struct::bitfield;

/// A guest segment's access-rights doubleword, as the VMCS lays it out.
///
/// The reserved runs — bits 11:8 and 31:17 — are unnamed bitfield members, so
/// they read back as zero and cannot be written by accident, the same
/// discipline `svm` applies to its reserved bits.
#[bitfield(u32)]
#[derive(PartialEq, Eq)]
pub struct AccessRights {
    /// The segment type: for data, expand-down and writable; for code,
    /// conforming and readable; for a system segment, which kind it is.
    #[bits(4)]
    pub kind: u8,
    /// Whether this is a code or data segment rather than a system one.
    pub descriptor: bool,
    /// The privilege level the descriptor was written with.
    #[bits(2)]
    pub dpl: u8,
    /// Whether the segment is present.
    pub present: bool,
    #[bits(4)]
    __: u8,
    /// The bit the architecture leaves entirely to software.
    pub available: bool,
    /// Whether this code segment runs in 64-bit mode.
    pub long: bool,
    /// The default operand and address size: set for 32-bit, clear for 16-bit.
    pub default_size: bool,
    /// Whether the descriptor's limit counted pages rather than bytes.
    pub granularity: bool,
    /// Whether the segment is unusable. Set for a segment the guest is not
    /// using, which is how VMX spells what a descriptor table spells with a
    /// null selector; a VM entry skips the consistency checks on an unusable
    /// segment.
    pub unusable: bool,
    #[bits(15)]
    __: u16,
}

#[cfg(test)]
mod tests {
    use super::AccessRights;

    #[test]
    fn a_typical_long_mode_code_segment_round_trips() {
        // Execute/read code (type 0xB), a code-or-data descriptor, ring 0,
        // present, long mode.
        let rights = AccessRights::new()
            .with_kind(0xB)
            .with_descriptor(true)
            .with_dpl(0)
            .with_present(true)
            .with_long(true);
        let raw = rights.into_bits();
        let back = AccessRights::from_bits(raw);
        assert_eq!(back.kind(), 0xB);
        assert!(back.descriptor());
        assert_eq!(back.dpl(), 0);
        assert!(back.present());
        assert!(back.long());
        assert!(!back.unusable());
    }

    #[test]
    fn the_unusable_bit_is_bit_sixteen() {
        assert_eq!(AccessRights::new().with_unusable(true).into_bits(), 1 << 16);
    }

    #[test]
    fn the_reserved_gap_above_present_stays_zero() {
        // Set every named field; bits 11:8 and 31:17 must remain clear.
        let rights = AccessRights::new()
            .with_kind(0xF)
            .with_descriptor(true)
            .with_dpl(3)
            .with_present(true)
            .with_available(true)
            .with_long(true)
            .with_default_size(true)
            .with_granularity(true)
            .with_unusable(true);
        let raw = rights.into_bits();
        assert_eq!(raw & 0x0000_0F00, 0, "bits 11:8 are reserved");
        assert_eq!(raw & 0xFFFE_0000, 0, "bits 31:17 are reserved");
    }
}
