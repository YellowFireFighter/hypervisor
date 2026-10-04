//! Building a guest's second translation out of the EPT entries `vmx` defines.
//!
//! This is the Intel counterpart to the `npt` crate: where `npt` builds AMD's
//! nested page tables, this builds Intel's extended page tables. It owns no
//! memory of its own — a caller supplies frames and a way to reach them through
//! the [`Memory`] trait — so the same builder runs over real frames in the
//! hypervisor and over ordinary allocations under test, which is what lets the
//! tree construction be checked on the host. An identity tree this crate built
//! has also been installed on an Intel processor and walked while it ran a
//! guest.
//!
//! Two ways to build a tree are offered. [`identity`] maps a run of
//! guest-physical memory straight through to the same host-physical addresses
//! with large pages, which is the mapping a guest that already uses the host's
//! physical layout needs; it is the whole of what enabling EPT for such a guest
//! takes. [`map`] places one 4-KiB guest page at an arbitrary host frame, for
//! the finer control a guest with its own physical layout needs.
//!
//! # The levels
//!
//! A guest-physical address is split the way an ordinary four-level walk splits
//! a virtual one: nine bits per level from bit 39 down, and twelve bits of
//! offset. A leaf at the lowest level maps a 4-KiB page; a leaf at the
//! page-directory level, which [`identity`] uses, maps a 2-MiB page, the large
//! page every EPT implementation supports.

#![no_std]

use vmx::{EptAccess, EptEntry, EptMemoryType, EptPointer};

/// Entries in one paging structure.
pub const ENTRIES: usize = 512;

/// How many levels the walk has.
const LEVELS: usize = 4;

/// The page-walk length written into the EPT pointer, which must match the
/// number of levels.
const WALK_LENGTH: u8 = 4;
const _: () = assert!(WALK_LENGTH as usize == LEVELS);

/// Bytes a leaf at the page-directory level maps.
const LARGE_PAGE: u64 = 2 * 1024 * 1024;

/// Bytes a leaf at the lowest level maps.
const SMALL_PAGE: u64 = 4096;

/// How many 4-KiB leaves a 2-MiB large page splits into.
const SPLIT_FANOUT: usize = (LARGE_PAGE / SMALL_PAGE) as usize;
const _: () = assert!(SPLIT_FANOUT == ENTRIES);

/// Bytes one page-directory of large pages covers: the whole of its 512 leaves.
const DIRECTORY_SPAN: u64 = LARGE_PAGE * ENTRIES as u64;

/// The lowest bit of each level's index, from the top of the walk down: the
/// page-map level 4, the page-directory-pointer table, the page directory, and
/// the page table.
const LEVEL_SHIFTS: [u32; LEVELS] = [39, 30, 21, 12];

/// Bits an index occupies once shifted down.
const INDEX_MASK: u64 = ENTRIES as u64 - 1;

/// What every intermediate entry grants: a pointer narrows nothing, so the
/// leaves below it decide the real access.
const ALL_ACCESS: EptAccess = EptAccess::READ
    .union(EptAccess::WRITE)
    .union(EptAccess::EXECUTE);

/// A source of EPT table frames and the means to reach one.
///
/// A frame is named by its physical address, which is what the entries store
/// and what the processor walks; [`table`](Self::table) turns that back into a
/// reference the builder writes through. In the hypervisor that reference comes
/// from the direct map; under test it comes from the allocation itself.
pub trait Memory {
    /// Allocates a zeroed paging structure and returns its physical address, or
    /// `None` when no frame is available.
    fn allocate(&mut self) -> Option<u64>;

    /// The paging structure at physical address `phys`, which must be one this
    /// `Memory` allocated.
    fn table(&mut self, phys: u64) -> &mut [EptEntry; ENTRIES];
}

/// Why a tree could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EptError {
    /// A frame was needed for a paging structure and none was available.
    OutOfFrames,
}

/// The index into the structure at `level` (zero at the top) for a
/// guest-physical address.
fn index(level: usize, guest_physical: u64) -> usize {
    ((guest_physical >> LEVEL_SHIFTS[level]) & INDEX_MASK) as usize
}

/// Builds an EPT that maps the first `gibibytes` GiB of guest-physical memory
/// to the same host-physical addresses, using 2-MiB pages, and returns the
/// pointer to install in the VMCS.
///
/// This is the mapping a guest that runs in the host's physical layout needs:
/// every guest-physical address is the host-physical address it already was,
/// so nothing moves, but the access now passes through EPT. Write-back is the
/// memory type, as for ordinary RAM.
///
/// # Errors
///
/// [`EptError::OutOfFrames`] if a paging structure cannot be allocated.
pub fn identity(memory: &mut impl Memory, gibibytes: usize) -> Result<EptPointer, EptError> {
    let pml4 = memory.allocate().ok_or(EptError::OutOfFrames)?;
    let pdpt = memory.allocate().ok_or(EptError::OutOfFrames)?;
    memory.table(pml4)[0] = EptEntry::table(pdpt, ALL_ACCESS);

    for gib in 0..gibibytes {
        let directory = memory.allocate().ok_or(EptError::OutOfFrames)?;
        memory.table(pdpt)[gib] = EptEntry::table(directory, ALL_ACCESS);
        let base = gib as u64 * DIRECTORY_SPAN;
        let table = memory.table(directory);
        for (slot, entry) in table.iter_mut().enumerate() {
            let address = base + slot as u64 * LARGE_PAGE;
            *entry = EptEntry::leaf(address, ALL_ACCESS, EptMemoryType::WriteBack, true);
        }
    }

    Ok(pointer(pml4))
}

/// The EPT pointer naming `root` as the top-level structure, with the standard
/// four-level walk and write-back paging, ready to install in a VMCS.
///
/// [`identity`] returns its own pointer; this is for a tree assembled with
/// [`map`], whose caller allocated the root and holds its address.
#[must_use]
pub fn pointer(root: u64) -> EptPointer {
    EptPointer::new(root, EptMemoryType::WriteBack, WALK_LENGTH, false)
}

/// Maps the 4-KiB guest-physical page at `guest_physical` to the host frame at
/// `host_physical`, with `access` and `memory_type`, building intermediate
/// tables under `root` as needed.
///
/// `root` is the physical address of the top-level structure, from a prior
/// [`Memory::allocate`]; a fresh tree starts by allocating one and mapping into
/// it. Both addresses must be 4-KiB aligned.
///
/// # Errors
///
/// [`EptError::OutOfFrames`] if an intermediate structure cannot be allocated.
pub fn map(
    memory: &mut impl Memory,
    root: u64,
    guest_physical: u64,
    host_physical: u64,
    access: EptAccess,
    memory_type: EptMemoryType,
) -> Result<(), EptError> {
    let mut structure = root;
    // Walk the three pointer levels, creating a child wherever the path does not
    // yet have one, splitting a large page where the path runs through one, and
    // descending into it.
    for level in 0..LEVELS - 1 {
        let slot = index(level, guest_physical);
        let existing = memory.table(structure)[slot];
        let child = if existing.is_present() && !existing.is_large() {
            existing.address()
        } else {
            let table = if existing.is_present() {
                split_large(memory, existing)?
            } else {
                memory.allocate().ok_or(EptError::OutOfFrames)?
            };
            memory.table(structure)[slot] = EptEntry::table(table, ALL_ACCESS);
            table
        };
        structure = child;
    }
    let slot = index(LEVELS - 1, guest_physical);
    memory.table(structure)[slot] = EptEntry::leaf(host_physical, access, memory_type, false);
    Ok(())
}

/// Splits a 2-MiB large page into a page table of 4-KiB leaves mapping the same
/// range with the same access and memory type, and returns the new table's
/// physical address.
///
/// Only the 2-MiB page-directory leaves [`identity`] builds are ever split; the
/// builder produces no 1-GiB leaves, so the children are always 4-KiB. The
/// split preserves the mapping exactly, so the caller can then place a finer
/// entry beneath it without disturbing the rest of the range.
fn split_large(memory: &mut impl Memory, large: EptEntry) -> Result<u64, EptError> {
    let table = memory.allocate().ok_or(EptError::OutOfFrames)?;
    let base = large.address();
    let access = large.access();
    let memory_type = large.memory_type().unwrap_or(EptMemoryType::WriteBack);
    for slot in 0..SPLIT_FANOUT {
        memory.table(table)[slot] =
            EptEntry::leaf(base + slot as u64 * SMALL_PAGE, access, memory_type, false);
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::{boxed::Box, vec::Vec};

    use vmx::{EptAccess, EptEntry, EptMemoryType};

    use super::{DIRECTORY_SPAN, ENTRIES, LARGE_PAGE, Memory, identity, index, map};

    /// Frames backed by ordinary allocations, with a synthetic physical address
    /// per frame so the builder's addresses round-trip.
    struct TestMemory {
        frames: Vec<Box<[EptEntry; ENTRIES]>>,
        base: u64,
    }

    impl TestMemory {
        fn new() -> Self {
            Self {
                frames: Vec::new(),
                // A 4-KiB-aligned base so leaf addresses keep their low bits
                // clear, as real frames do.
                base: 0x10_0000,
            }
        }

        fn slot(&self, phys: u64) -> usize {
            ((phys - self.base) / 0x1000) as usize
        }
    }

    impl Memory for TestMemory {
        fn allocate(&mut self) -> Option<u64> {
            let index = self.frames.len();
            self.frames.push(Box::new([EptEntry::ABSENT; ENTRIES]));
            Some(self.base + index as u64 * 0x1000)
        }

        fn table(&mut self, phys: u64) -> &mut [EptEntry; ENTRIES] {
            let index = self.slot(phys);
            &mut self.frames[index]
        }
    }

    #[test]
    fn an_address_splits_into_the_four_level_indices() {
        // One bit set in each level's field.
        let guest = (1 << 39) | (2 << 30) | (3 << 21) | (4 << 12);
        assert_eq!(index(0, guest), 1);
        assert_eq!(index(1, guest), 2);
        assert_eq!(index(2, guest), 3);
        assert_eq!(index(3, guest), 4);
    }

    #[test]
    fn identity_maps_each_large_page_to_itself() {
        let mut memory = TestMemory::new();
        let eptp = identity(&mut memory, 2).expect("frames are unbounded in the test");
        assert_eq!(eptp.levels(), 4);
        assert_eq!(eptp.memory_type(), Some(EptMemoryType::WriteBack));

        // Walk to a chosen 2-MiB page and confirm it maps to itself.
        let target = DIRECTORY_SPAN + 5 * LARGE_PAGE; // GiB 1, page 5
        let pml4 = eptp.root();
        let pdpt = memory.table(pml4)[0].address();
        let directory = memory.table(pdpt)[1].address();
        let leaf = memory.table(directory)[5];
        assert_eq!(leaf.address(), target);
        assert!(leaf.is_large());
        assert_eq!(leaf.memory_type(), Some(EptMemoryType::WriteBack));
    }

    #[test]
    fn identity_covers_the_whole_requested_range() {
        let mut memory = TestMemory::new();
        let eptp = identity(&mut memory, 3).expect("frames are unbounded");
        let pml4 = eptp.root();
        let pdpt = memory.table(pml4)[0].address();
        // Three page directories are present, one per GiB.
        for gib in 0..3 {
            assert!(memory.table(pdpt)[gib].is_present(), "GiB {gib} unmapped");
        }
        assert!(
            !memory.table(pdpt)[3].is_present(),
            "GiB 3 should be absent"
        );
    }

    #[test]
    fn map_places_one_page_and_builds_the_path_to_it() {
        let mut memory = TestMemory::new();
        let root = memory.allocate().expect("a root frame");
        let guest = (7 << 39) | (6 << 30) | (5 << 21) | (4 << 12);
        let host = 0x4000;
        map(
            &mut memory,
            root,
            guest,
            host,
            EptAccess::READ | EptAccess::EXECUTE,
            EptMemoryType::WriteBack,
        )
        .expect("frames are unbounded");

        let pdpt = memory.table(root)[7].address();
        let directory = memory.table(pdpt)[6].address();
        let table = memory.table(directory)[5].address();
        let leaf = memory.table(table)[4];
        assert_eq!(leaf.address(), host);
        assert!(!leaf.is_large());
        assert_eq!(leaf.access(), EptAccess::READ | EptAccess::EXECUTE);
    }

    #[test]
    fn mapping_a_page_inside_a_large_identity_page_splits_it() {
        // Build an identity map, then place one 4-KiB page over a frame inside
        // an existing 2-MiB large page. The large page must split: the target
        // frame gets the finer entry, and a sibling 4-KiB frame in the same
        // 2-MiB page still maps to itself, so the rest of the range is intact.
        let mut memory = TestMemory::new();
        let eptp = identity(&mut memory, 2).expect("frames are unbounded in the test");
        let large_base = DIRECTORY_SPAN + 5 * LARGE_PAGE;
        let target = large_base + 7 * 0x1000;
        let sibling = large_base + 9 * 0x1000;
        let rwx = EptAccess::READ | EptAccess::WRITE | EptAccess::EXECUTE;
        map(
            &mut memory,
            eptp.root(),
            target,
            target,
            rwx,
            EptMemoryType::Uncacheable,
        )
        .expect("frames are unbounded");

        let pdpt = memory.table(eptp.root())[0].address();
        let directory = memory.table(pdpt)[1].address();
        let table_entry = memory.table(directory)[5];
        assert!(!table_entry.is_large(), "the large page did not split");
        let table = table_entry.address();
        let placed = memory.table(table)[7];
        assert_eq!(placed.address(), target);
        assert!(!placed.is_large());
        assert_eq!(placed.memory_type(), Some(EptMemoryType::Uncacheable));
        let kept = memory.table(table)[9];
        assert_eq!(kept.address(), sibling, "the split lost a sibling page");
        assert_eq!(kept.memory_type(), Some(EptMemoryType::WriteBack));
    }

    #[test]
    fn mapping_two_pages_in_one_directory_shares_the_path() {
        let mut memory = TestMemory::new();
        let root = memory.allocate().expect("a root frame");
        let rwx = EptAccess::READ | EptAccess::WRITE | EptAccess::EXECUTE;
        // Two pages that differ only in the page-table index share every higher
        // structure, so the second map allocates no new intermediate table.
        map(
            &mut memory,
            root,
            0x1000,
            0xA000,
            rwx,
            EptMemoryType::WriteBack,
        )
        .unwrap();
        let after_first = memory.frames.len();
        map(
            &mut memory,
            root,
            0x2000,
            0xB000,
            rwx,
            EptMemoryType::WriteBack,
        )
        .unwrap();
        assert_eq!(
            memory.frames.len(),
            after_first,
            "the second page should reuse the first's tables"
        );
        let pdpt = memory.table(root)[0].address();
        let directory = memory.table(pdpt)[0].address();
        let table = memory.table(directory)[0].address();
        assert_eq!(memory.table(table)[1].address(), 0xA000);
        assert_eq!(memory.table(table)[2].address(), 0xB000);
    }
}
