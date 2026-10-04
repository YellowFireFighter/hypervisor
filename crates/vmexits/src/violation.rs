//! Decoding an EPT violation from its exit qualification.
//!
//! When a guest's access is forbidden by the extended page tables, the
//! processor describes it in the exit qualification: which kind of access the
//! guest attempted — read, write or instruction fetch — and which of those the
//! entry that governed the page actually granted. Reading those bits out is
//! pure, so it is stated and tested here.
//!
//! Both halves are expressed as an [`EptAccess`], the same read/write/execute
//! grant the `ept` tree is built from: the attempted access occupies the low
//! three bits of the qualification and the granted access the three above it,
//! in the same order, so each is one mask of that type. A fetch is mapped to
//! execute, the grant it would need.
//!
//! No handler maps a guest's own memory yet, so a violation is reported rather
//! than resolved; the decode is what makes the report say precisely what the
//! guest was denied.

use vmx::EptAccess;

/// An EPT violation, as the exit qualification describes it.
///
/// A violation is the [`attempted`](Self::attempted) access of something the
/// [`granted`](Self::granted) access did not allow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Violation {
    /// What the guest tried to do: some combination of read, write and execute
    /// (an instruction fetch).
    pub attempted: EptAccess,
    /// What the governing entry allowed.
    pub granted: EptAccess,
}

/// The attempted access is the low three bits — read, write, fetch — which are
/// the read, write and execute positions of an [`EptAccess`].
const ACCESS_MASK: u64 = 0b111;
/// The granted access sits three bits above the attempted one.
const GRANTED_SHIFT: u64 = 3;

impl Violation {
    /// Takes an EPT-violation exit qualification apart.
    #[must_use]
    pub const fn decode(qualification: u64) -> Self {
        Self {
            attempted: EptAccess::from_bits_truncate(qualification & ACCESS_MASK),
            granted: EptAccess::from_bits_truncate((qualification >> GRANTED_SHIFT) & ACCESS_MASK),
        }
    }
}

#[cfg(test)]
mod tests {
    use vmx::EptAccess;

    use super::{GRANTED_SHIFT, Violation};

    #[test]
    fn a_write_to_a_read_only_page_decodes() {
        // Attempted write (bit 1), on an entry granting read (bit 3) but not
        // write.
        let violation = Violation::decode((1 << 1) | (1 << 3));
        assert_eq!(violation.attempted, EptAccess::WRITE);
        assert_eq!(violation.granted, EptAccess::READ);
    }

    #[test]
    fn a_fetch_from_a_no_execute_page_decodes() {
        // Attempted fetch (bit 2), on an entry granting read and write but not
        // execute.
        let violation = Violation::decode((1 << 2) | (1 << 3) | (1 << 4));
        assert_eq!(violation.attempted, EptAccess::EXECUTE);
        assert_eq!(violation.granted, EptAccess::READ | EptAccess::WRITE);
        assert!(!violation.granted.contains(EptAccess::EXECUTE));
    }

    #[test]
    fn an_empty_qualification_is_all_clear() {
        let violation = Violation::decode(0);
        assert_eq!(violation.attempted, EptAccess::empty());
        assert_eq!(violation.granted, EptAccess::empty());
    }

    #[test]
    fn the_granted_half_does_not_bleed_into_the_attempted_one() {
        // Only the granted bits (3:5) are set; the attempted half must be empty.
        let violation = Violation::decode(0b111 << GRANTED_SHIFT);
        assert_eq!(violation.attempted, EptAccess::empty());
        assert_eq!(
            violation.granted,
            EptAccess::READ | EptAccess::WRITE | EptAccess::EXECUTE
        );
    }
}
