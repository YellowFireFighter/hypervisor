//! `IA32_VMX_BASIC`, the register that describes this processor's VMCS.
//!
//! Before a hypervisor allocates a VMCS it has to know three things the
//! architecture does not fix: the revision identifier to stamp into the first
//! doubleword of every VMCS and VMXON region, how many bytes those regions
//! occupy, and the memory type the processor expects to access them with. All
//! three come from this one read-only model-specific register, so it is read
//! once at bring-up and its answer settles how the regions are allocated.
//!
//! As everywhere in this crate, nothing here reads the register — a caller
//! brings the 64-bit value and this says what it means.

use core::fmt::{self, Debug, Formatter};

/// The number of this register.
pub const IA32_VMX_BASIC: u32 = 0x480;

/// The revision identifier occupies the low 31 bits.
const REVISION_MASK: u64 = 0x7FFF_FFFF;
/// The VMCS/VMXON region size is in bits 44:32.
const SIZE_SHIFT: u64 = 32;
/// The region size is thirteen bits wide.
const SIZE_MASK: u64 = 0x1FFF;
/// The memory-type code is in bits 53:50.
const MEMORY_TYPE_SHIFT: u64 = 50;
/// The memory-type code is four bits wide.
const MEMORY_TYPE_MASK: u64 = 0xF;
/// Bit 55 reports that the "true" capability registers are present.
const TRUE_CONTROLS_BIT: u64 = 1 << 55;

/// The memory type the processor accesses a VMCS and its referenced structures
/// with.
///
/// Only two of the sixteen codes ever appear here: a processor reports
/// write-back on every part pulzar would run on, and uncacheable on some early
/// or constrained parts. The rest are reserved, so [`MemoryType::from_code`]
/// answers `None` for them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    /// Uncacheable.
    Uncacheable,
    /// Write-back, which is what a VMCS is normally accessed with.
    WriteBack,
}

impl MemoryType {
    /// The memory type a four-bit code names, or `None` for a reserved code.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Uncacheable),
            6 => Some(Self::WriteBack),
            _ => None,
        }
    }
}

/// A decoded `IA32_VMX_BASIC`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct VmxBasic(u64);

impl VmxBasic {
    /// Takes the register's raw value apart.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The revision identifier to stamp into the first doubleword of every
    /// VMCS and VMXON region this processor is given.
    #[must_use]
    pub const fn revision(self) -> u32 {
        (self.0 & REVISION_MASK) as u32
    }

    /// How many bytes a VMCS and the VMXON region occupy.
    ///
    /// Never more than [`PAGE_BYTES`](crate::PAGE_BYTES), so the region a
    /// caller allocates is a page and this is how much of it the processor
    /// uses.
    #[must_use]
    pub const fn region_bytes(self) -> u32 {
        ((self.0 >> SIZE_SHIFT) & SIZE_MASK) as u32
    }

    /// The memory type the processor accesses these regions with, or `None`
    /// if the register reports a reserved code — which a sound processor does
    /// not, and which a caller treats as a refusal to enter VMX rather than a
    /// guess.
    #[must_use]
    pub const fn memory_type(self) -> Option<MemoryType> {
        MemoryType::from_code(((self.0 >> MEMORY_TYPE_SHIFT) & MEMORY_TYPE_MASK) as u8)
    }

    /// Whether the processor provides the `IA32_VMX_TRUE_*` capability
    /// registers, which report the control bits that are genuinely flexible
    /// rather than those a default class forces.
    #[must_use]
    pub const fn has_true_controls(self) -> bool {
        self.0 & TRUE_CONTROLS_BIT != 0
    }
}

impl Debug for VmxBasic {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VmxBasic")
            .field("revision", &self.revision())
            .field("region_bytes", &self.region_bytes())
            .field("memory_type", &self.memory_type())
            .field("true_controls", &self.has_true_controls())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{IA32_VMX_BASIC, MemoryType, VmxBasic};
    use crate::PAGE_BYTES;

    #[test]
    fn the_register_number_is_the_architectural_one() {
        assert_eq!(IA32_VMX_BASIC, 0x480);
    }

    #[test]
    fn a_typical_write_back_value_decodes() {
        // Revision 1, a 4096-byte region, write-back memory, true controls.
        let bits = 1 | (4096 << 32) | (6 << 50) | (1 << 55);
        let basic = VmxBasic::from_bits(bits);
        assert_eq!(basic.revision(), 1);
        assert_eq!(basic.region_bytes() as usize, PAGE_BYTES);
        assert_eq!(basic.memory_type(), Some(MemoryType::WriteBack));
        assert!(basic.has_true_controls());
    }

    #[test]
    fn the_revision_ignores_the_bits_above_it() {
        // Bit 31 is reserved zero; set everything above the revision and check
        // the revision still reads only its own 31 bits.
        let basic = VmxBasic::from_bits(!0);
        assert_eq!(basic.revision(), 0x7FFF_FFFF);
    }

    #[test]
    fn a_reserved_memory_type_is_no_memory_type() {
        let basic = VmxBasic::from_bits(0xF << 50);
        assert_eq!(basic.memory_type(), None);
    }
}
