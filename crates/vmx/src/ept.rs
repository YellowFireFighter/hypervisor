//! Extended page tables: the second translation a guest's memory passes
//! through on Intel.
//!
//! This is the VMX counterpart to the `npt` crate's entry format, and the
//! important thing about it is that it is *not* the ordinary page-table format.
//! AMD's nested page tables are x86-64 page tables with the familiar bits, so
//! `npt` reuses that format. Intel's extended page tables have their own: the
//! permission bits sit where the present, writable and no-execute bits do not,
//! read and write and execute are three separate grants rather than a present
//! bit and a writable bit, and a leaf entry carries an EPT memory type in a
//! field the ordinary format has no equivalent of. Stating that format here is
//! what lets a future EPT mapper be written against named bits rather than
//! against the raw numbers.
//!
//! As with the rest of the crate this is layout only: building a tree of these
//! entries, pointing the processor at it and shooting down its cached
//! translations are a mapper's work, not this module's.

use bitflags::bitflags;

/// Bits 51:12 of an entry or pointer: the physical address of the next
/// structure or the page frame, which is page-aligned.
const ADDRESS_MASK: u64 = 0x000F_FFFF_FFFF_F000;

bitflags! {
    /// The access an EPT entry grants to the guest-physical range it covers.
    ///
    /// Read, write and execute are independent, unlike the ordinary format's
    /// present-and-writable pair: an entry with no access bits set is not
    /// present, and an EPT violation reports which access was attempted against
    /// what was granted.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct EptAccess: u64 {
        /// The guest may read the range.
        const READ = 1 << 0;
        /// The guest may write the range.
        const WRITE = 1 << 1;
        /// The guest may execute from the range. With mode-based execute
        /// control this is supervisor-mode execution; without it, all
        /// execution.
        const EXECUTE = 1 << 2;
        /// The guest may execute from the range in user mode, honoured only
        /// with mode-based execute control.
        const USER_EXECUTE = 1 << 10;
    }
}

/// The memory type an EPT leaf entry assigns to the range it maps.
///
/// The codes are the architecture's, which are not the same set as the VMCS
/// memory type in [`basic`](crate::basic): EPT leaves can name write-combining,
/// write-through and write-protected ranges a VMCS access never would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EptMemoryType {
    /// Uncacheable.
    Uncacheable = 0,
    /// Write-combining.
    WriteCombining = 1,
    /// Write-through.
    WriteThrough = 4,
    /// Write-protected.
    WriteProtected = 5,
    /// Write-back, the type ordinary guest RAM is mapped with.
    WriteBack = 6,
}

impl EptMemoryType {
    /// The three-bit code for this memory type.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The memory type a three-bit code names, or `None` for a reserved code.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Uncacheable),
            1 => Some(Self::WriteCombining),
            4 => Some(Self::WriteThrough),
            5 => Some(Self::WriteProtected),
            6 => Some(Self::WriteBack),
            _ => None,
        }
    }
}

/// The memory-type field of a leaf entry begins at bit 3.
const MEMORY_TYPE_SHIFT: u64 = 3;
/// The memory-type field is three bits wide.
const MEMORY_TYPE_MASK: u64 = 0b111;
/// Bit 6 tells the processor to ignore the guest's PAT for this leaf.
const IGNORE_PAT_BIT: u64 = 1 << 6;
/// Bit 7 marks a `PDPTE` or `PDE` a leaf mapping a large page rather than a
/// pointer to the next table.
const LARGE_PAGE_BIT: u64 = 1 << 7;

/// One extended-page-table entry: either a pointer to the next structure or a
/// leaf mapping a page of guest-physical memory.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct EptEntry(u64);

impl EptEntry {
    /// An absent entry: no access, so the processor walks no further and
    /// reports an EPT violation on any access to the range.
    pub const ABSENT: Self = Self(0);

    /// A non-leaf entry pointing at the next paging structure at `next`, which
    /// must be page-aligned, granting `access`.
    ///
    /// An intermediate entry's access bits bound what its subtree can grant,
    /// so a pointer is normally given read, write and execute and the leaves
    /// below it narrow that.
    #[must_use]
    pub const fn table(next: u64, access: EptAccess) -> Self {
        Self((next & ADDRESS_MASK) | access.bits())
    }

    /// A leaf entry mapping the page frame at `frame`, which must be aligned
    /// to the leaf's size, granting `access` and assigning `memory_type`.
    ///
    /// `large` is set for a leaf at the page-directory or
    /// page-directory-pointer level, which maps a 2-MiB or 1-GiB page; a
    /// leaf at the lowest level leaves it clear.
    #[must_use]
    pub const fn leaf(
        frame: u64,
        access: EptAccess,
        memory_type: EptMemoryType,
        large: bool,
    ) -> Self {
        let mut bits = (frame & ADDRESS_MASK)
            | access.bits()
            | ((memory_type.code() as u64) << MEMORY_TYPE_SHIFT);
        if large {
            bits |= LARGE_PAGE_BIT;
        }
        Self(bits)
    }

    /// The raw entry.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// The access this entry grants.
    #[must_use]
    pub const fn access(self) -> EptAccess {
        EptAccess::from_bits_truncate(self.0)
    }

    /// Whether this entry grants any access at all; a fully absent entry does
    /// not, and the processor stops the walk there.
    #[must_use]
    pub const fn is_present(self) -> bool {
        self.0 & (EptAccess::READ.bits() | EptAccess::WRITE.bits() | EptAccess::EXECUTE.bits()) != 0
    }

    /// Whether this entry is a leaf mapping a large page rather than a pointer
    /// to the next structure.
    #[must_use]
    pub const fn is_large(self) -> bool {
        self.0 & LARGE_PAGE_BIT != 0
    }

    /// The physical address of the next structure or the page frame.
    #[must_use]
    pub const fn address(self) -> u64 {
        self.0 & ADDRESS_MASK
    }

    /// The memory type a leaf entry assigns, or `None` for a reserved code.
    ///
    /// Meaningful only on a leaf; the field overlaps bits a pointer entry uses
    /// for nothing, so a caller reads it only where
    /// [`is_large`](Self::is_large) or the leaf level already says this
    /// entry maps a page.
    #[must_use]
    pub const fn memory_type(self) -> Option<EptMemoryType> {
        EptMemoryType::from_code(((self.0 >> MEMORY_TYPE_SHIFT) & MEMORY_TYPE_MASK) as u8)
    }

    /// This entry with the guest's PAT ignored for the range, meaningful only
    /// on a leaf.
    #[must_use]
    pub const fn ignoring_pat(self) -> Self {
        Self(self.0 | IGNORE_PAT_BIT)
    }
}

/// The memory type begins at bit 0 of the EPT pointer.
const EPTP_MEMORY_TYPE_MASK: u64 = 0b111;
/// The page-walk-length-minus-one field begins at bit 3.
const EPTP_WALK_SHIFT: u64 = 3;
/// The page-walk-length field is three bits wide.
const EPTP_WALK_MASK: u64 = 0b111;
/// Bit 6 enables the accessed and dirty flags in the EPT entries.
const EPTP_ACCESSED_DIRTY_BIT: u64 = 1 << 6;

/// The extended-page-table pointer: the root of the second translation, as it
/// is written into the VMCS [`EPT_POINTER`](crate::Field::EPT_POINTER) field.
///
/// It names the top-level table, the memory type its structures are walked
/// with, and how many levels the walk has.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct EptPointer(u64);

impl EptPointer {
    /// A pointer to the top-level table at `root`, walked as `levels` levels
    /// with structures accessed as `memory_type`, enabling the accessed and
    /// dirty flags iff `accessed_dirty` is set.
    ///
    /// `levels` is four for the ordinary four-level walk; it is stored as the
    /// architecture's length-minus-one.
    #[must_use]
    pub const fn new(
        root: u64,
        memory_type: EptMemoryType,
        levels: u8,
        accessed_dirty: bool,
    ) -> Self {
        let mut bits = (root & ADDRESS_MASK)
            | (memory_type.code() as u64 & EPTP_MEMORY_TYPE_MASK)
            | ((levels.saturating_sub(1) as u64 & EPTP_WALK_MASK) << EPTP_WALK_SHIFT);
        if accessed_dirty {
            bits |= EPTP_ACCESSED_DIRTY_BIT;
        }
        Self(bits)
    }

    /// The raw pointer, as written into the VMCS.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// The physical address of the top-level table.
    #[must_use]
    pub const fn root(self) -> u64 {
        self.0 & ADDRESS_MASK
    }

    /// How many levels the page walk has.
    #[must_use]
    pub const fn levels(self) -> u8 {
        (((self.0 >> EPTP_WALK_SHIFT) & EPTP_WALK_MASK) as u8) + 1
    }

    /// The memory type the EPT structures are accessed with, or `None` if the
    /// field holds a reserved code.
    #[must_use]
    pub const fn memory_type(self) -> Option<EptMemoryType> {
        EptMemoryType::from_code((self.0 & EPTP_MEMORY_TYPE_MASK) as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::{EptAccess, EptEntry, EptMemoryType, EptPointer};

    #[test]
    fn a_memory_type_round_trips_through_its_code() {
        for ty in [
            EptMemoryType::Uncacheable,
            EptMemoryType::WriteCombining,
            EptMemoryType::WriteThrough,
            EptMemoryType::WriteProtected,
            EptMemoryType::WriteBack,
        ] {
            assert_eq!(EptMemoryType::from_code(ty.code()), Some(ty));
        }
        assert_eq!(EptMemoryType::from_code(2), None);
        assert_eq!(EptMemoryType::from_code(3), None);
        assert_eq!(EptMemoryType::from_code(7), None);
    }

    #[test]
    fn an_absent_entry_grants_nothing_and_is_not_present() {
        assert_eq!(EptEntry::ABSENT.bits(), 0);
        assert!(!EptEntry::ABSENT.is_present());
    }

    #[test]
    fn a_table_entry_keeps_its_address_and_access() {
        let entry = EptEntry::table(0x1234_5000, EptAccess::READ | EptAccess::WRITE);
        assert_eq!(entry.address(), 0x1234_5000);
        assert_eq!(entry.access(), EptAccess::READ | EptAccess::WRITE);
        assert!(entry.is_present());
        assert!(!entry.is_large());
    }

    #[test]
    fn a_write_back_leaf_carries_its_frame_access_type_and_size() {
        let all = EptAccess::READ | EptAccess::WRITE | EptAccess::EXECUTE;
        let entry = EptEntry::leaf(0x20_0000, all, EptMemoryType::WriteBack, true);
        assert_eq!(entry.address(), 0x20_0000);
        assert_eq!(entry.access(), all);
        assert!(entry.is_large());
        assert_eq!(entry.memory_type(), Some(EptMemoryType::WriteBack));
        // Memory type 6 sits in bits 5:3.
        assert_eq!((entry.bits() >> 3) & 0b111, 6);
    }

    #[test]
    fn a_pointer_encodes_the_root_levels_and_memory_type() {
        let eptp = EptPointer::new(0xAB_C000, EptMemoryType::WriteBack, 4, true);
        assert_eq!(eptp.root(), 0xAB_C000);
        assert_eq!(eptp.levels(), 4);
        assert_eq!(eptp.memory_type(), Some(EptMemoryType::WriteBack));
        // Walk length minus one is 3, in bits 5:3.
        assert_eq!((eptp.bits() >> 3) & 0b111, 3);
        // The accessed/dirty bit is set.
        assert_eq!(eptp.bits() & (1 << 6), 1 << 6);
    }
}
