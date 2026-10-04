//! The VMXON region and the VMCS, as far as their format is software's to set.
//!
//! VMX has two processor-owned pages whose contents are otherwise opaque. The
//! VMXON region is handed to `VMXON` to enter VMX operation; a VMCS is handed
//! to `VMPTRLD` to make it current. Each is [`PAGE_BYTES`](crate::PAGE_BYTES)
//! and each begins with one doubleword that *is* software's to set: the VMCS
//! revision identifier from [`VmxBasic::revision`](crate::VmxBasic::revision),
//! which stamps the page as belonging to this processor's VMX implementation.
//! Everything past that doubleword is the processor's own format, touched only
//! through `VMREAD` and `VMWRITE`, never by offset.
//!
//! For a VMCS the top bit of that doubleword is the *shadow-VMCS indicator*: a
//! shadow VMCS, used for nested virtualization, sets it; an ordinary VMCS and
//! every VMXON region leave it clear. This module computes those first
//! doublewords and nothing else — allocating and zeroing the page, and
//! executing the instructions, are a caller's work.

/// The shadow-VMCS indicator, the top bit of a VMCS's first doubleword.
const SHADOW_INDICATOR: u32 = 1 << 31;

/// The first doubleword of a VMXON region: the revision identifier with the
/// top bit clear.
///
/// The revision comes from `IA32_VMX_BASIC` and never has its top bit set, so
/// this is the revision unchanged; it is a named function rather than the bare
/// value so the VMXON and VMCS cases read alike at the call site.
#[must_use]
pub const fn vmxon_header(revision: u32) -> u32 {
    revision & !SHADOW_INDICATOR
}

/// The first doubleword of an ordinary (non-shadow) VMCS: the revision
/// identifier with the shadow indicator clear.
#[must_use]
pub const fn vmcs_header(revision: u32) -> u32 {
    revision & !SHADOW_INDICATOR
}

/// The first doubleword of a shadow VMCS: the revision identifier with the
/// shadow indicator set.
#[must_use]
pub const fn shadow_vmcs_header(revision: u32) -> u32 {
    (revision & !SHADOW_INDICATOR) | SHADOW_INDICATOR
}

/// Whether a VMCS's first doubleword marks it a shadow VMCS.
#[must_use]
pub const fn is_shadow(header: u32) -> bool {
    header & SHADOW_INDICATOR != 0
}

/// The revision identifier a region header carries, with the shadow indicator
/// masked off.
#[must_use]
pub const fn revision_of(header: u32) -> u32 {
    header & !SHADOW_INDICATOR
}

#[cfg(test)]
mod tests {
    use super::{is_shadow, revision_of, shadow_vmcs_header, vmcs_header, vmxon_header};

    #[test]
    fn a_plain_region_header_is_the_revision_unchanged() {
        assert_eq!(vmxon_header(1), 1);
        assert_eq!(vmcs_header(0x1234), 0x1234);
        assert!(!is_shadow(vmcs_header(0x1234)));
    }

    #[test]
    fn a_shadow_header_sets_the_top_bit_and_keeps_the_revision() {
        let header = shadow_vmcs_header(0x1234);
        assert!(is_shadow(header));
        assert_eq!(revision_of(header), 0x1234);
    }

    #[test]
    fn a_revision_with_the_top_bit_set_is_never_treated_as_shadow() {
        // IA32_VMX_BASIC never sets bit 31, but a header built from a stray
        // value should still report the revision without its top bit.
        assert_eq!(revision_of(vmxon_header(0xFFFF_FFFF)), 0x7FFF_FFFF);
        assert!(!is_shadow(vmxon_header(0xFFFF_FFFF)));
    }
}
