//! What is handed to a guest on the way in, and what it was handed when it came
//! out.
//!
//! The VMX counterpart to `svm`'s event injection. One field, the VM-entry
//! interruption-information field, describes an event to deliver on the next
//! entry; a field of the same shape, the VM-exit interruption-information
//! field, describes the event that was being delivered when the guest exited.
//! Both pack a vector, a type, whether an error code accompanies it, and a
//! valid bit into one doubleword, so they are built and read the same way and
//! modelled here once.

use core::fmt::{self, Debug, Formatter};

/// What kind of event an interruption-information field describes.
///
/// The value is the three-bit type code the architecture puts in bits 10:8, so
/// it round-trips through [`Interruption`] unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// A maskable external interrupt.
    External = 0,
    /// A non-maskable interrupt.
    Nmi = 2,
    /// A hardware exception: a fault, trap or abort the processor raised.
    HardwareException = 3,
    /// A software interrupt raised by `INT n`.
    SoftwareInterrupt = 4,
    /// A privileged software exception, from `INT1`.
    PrivilegedSoftwareException = 5,
    /// A software exception, from `INT3` or `INTO`.
    SoftwareException = 6,
}

impl Kind {
    /// The kind a three-bit type code names, or `None` for a reserved code.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::External),
            2 => Some(Self::Nmi),
            3 => Some(Self::HardwareException),
            4 => Some(Self::SoftwareInterrupt),
            5 => Some(Self::PrivilegedSoftwareException),
            6 => Some(Self::SoftwareException),
            _ => None,
        }
    }

    /// The three-bit type code for this kind.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// A VM-entry or VM-exit interruption-information field.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Interruption(u32);

/// The vector is the low eight bits.
const VECTOR_MASK: u32 = 0xFF;
/// The type code begins at bit 8.
const KIND_SHIFT: u32 = 8;
/// The type code is three bits wide.
const KIND_MASK: u32 = 0b111;
/// Bit 11 says an error code accompanies the event.
const ERROR_CODE_BIT: u32 = 1 << 11;
/// Bit 31 marks the field valid.
const VALID_BIT: u32 = 1 << 31;

impl Interruption {
    /// Takes an interruption-information field's raw value apart.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The field for injecting `vector` of `kind`, with an error code iff
    /// `error_code` is set, marked valid.
    #[must_use]
    pub const fn inject(vector: u8, kind: Kind, error_code: bool) -> Self {
        let mut bits = (vector as u32) | ((kind.code() as u32) << KIND_SHIFT) | VALID_BIT;
        if error_code {
            bits |= ERROR_CODE_BIT;
        }
        Self(bits)
    }

    /// The raw field.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether the field describes an event at all. A field whose valid bit is
    /// clear carries nothing, which on an exit means no event was in delivery.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 & VALID_BIT != 0
    }

    /// The vector of the event.
    #[must_use]
    pub const fn vector(self) -> u8 {
        (self.0 & VECTOR_MASK) as u8
    }

    /// The kind of the event, or `None` for a reserved type code.
    #[must_use]
    pub const fn kind(self) -> Option<Kind> {
        Kind::from_code(((self.0 >> KIND_SHIFT) & KIND_MASK) as u8)
    }

    /// Whether an error code accompanies the event.
    #[must_use]
    pub const fn has_error_code(self) -> bool {
        self.0 & ERROR_CODE_BIT != 0
    }
}

impl Debug for Interruption {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Interruption")
            .field("valid", &self.is_valid())
            .field("vector", &self.vector())
            .field("kind", &self.kind())
            .field("error_code", &self.has_error_code())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{Interruption, Kind};

    #[test]
    fn a_kind_round_trips_through_its_code() {
        for kind in [
            Kind::External,
            Kind::Nmi,
            Kind::HardwareException,
            Kind::SoftwareInterrupt,
            Kind::PrivilegedSoftwareException,
            Kind::SoftwareException,
        ] {
            assert_eq!(Kind::from_code(kind.code()), Some(kind));
        }
        assert_eq!(Kind::from_code(1), None);
        assert_eq!(Kind::from_code(7), None);
    }

    #[test]
    fn injecting_a_page_fault_sets_vector_kind_error_code_and_valid() {
        // A page fault is vector 14, a hardware exception, and carries an
        // error code.
        let event = Interruption::inject(14, Kind::HardwareException, true);
        assert!(event.is_valid());
        assert_eq!(event.vector(), 14);
        assert_eq!(event.kind(), Some(Kind::HardwareException));
        assert!(event.has_error_code());
        // Bits: vector 14, type 3 at bit 8, error-code bit 11, valid bit 31.
        assert_eq!(event.bits(), 0x0E | (3 << 8) | (1 << 11) | (1 << 31));
    }

    #[test]
    fn an_invalid_field_carries_no_event() {
        let none = Interruption::from_bits(0);
        assert!(!none.is_valid());
    }

    #[test]
    fn a_raw_field_decodes_to_the_parts_it_was_built_from() {
        let built = Interruption::inject(13, Kind::External, false);
        let raw = Interruption::from_bits(built.bits());
        assert_eq!(raw.vector(), 13);
        assert_eq!(raw.kind(), Some(Kind::External));
        assert!(!raw.has_error_code());
        assert!(raw.is_valid());
    }
}
