//! Writing a minimal 64-bit guest into the current VMCS.
//!
//! This programs the smallest guest that is a valid entry: one that runs in the
//! host's own address space, so no second translation is needed to reach the
//! code it executes. Its control registers, `EFER` and paging are the host's —
//! `GUEST_CR3` is the host's `CR3`, so a guest-linear address walks the host's
//! page tables — and its segments are flat: bases zero, limits the whole space,
//! a 64-bit code segment and writable data segments. The task register and the
//! descriptor-table bases are the host's, so the guest has somewhere valid to
//! fault to; the local descriptor table is marked unusable, as a guest that
//! never loads one has none.
//!
//! It is the counterpart to seeding a VMCB's save area, and it exists to be
//! entered: with a `RIP` pointing at an instruction that exits — `CPUID`,
//! unconditionally — it is enough to prove the world switch and the entry
//! checks, which is what the self-test uses it for. A guest that runs an
//! operating system needs far more than this, above all a second translation;
//! this is the floor, not the shape of the real thing.

use vmx::Field;
use x86_64::{
    instructions::tables::{sgdt, sidt},
    registers::control::{Cr0, Cr4},
};

use crate::{
    Vmcs,
    error::VmFail,
    host,
    msr::{self, IA32_EFER},
};

/// Access rights of a present 64-bit code segment: execute/read/accessed, a
/// code-or-data descriptor, ring 0, long mode, page granularity.
const CODE_ACCESS_RIGHTS: u64 = 0xA09B;
/// Access rights of a present writable data segment: read/write/accessed, a
/// code-or-data descriptor, ring 0, 32-bit default size, page granularity.
const DATA_ACCESS_RIGHTS: u64 = 0xC093;
/// Access rights of a present 64-bit busy task-state segment.
const TASK_ACCESS_RIGHTS: u64 = 0x8B;
/// Access rights marking a segment unusable, which is how a guest with no local
/// descriptor table describes it.
const UNUSABLE_ACCESS_RIGHTS: u64 = 0x1_0000;

/// A flat segment's limit: the whole address space, in pages.
const FLAT_LIMIT: u64 = 0xFFFF_FFFF;
/// The task-state segment's limit, the minimum a 64-bit TSS is.
const TASK_LIMIT: u64 = 0x67;

/// A code selector, index one, ring zero.
const CODE_SELECTOR: u64 = 0x08;
/// A data selector, index two, ring zero.
const DATA_SELECTOR: u64 = 0x10;

/// `RFLAGS` with only the reserved bit set: interrupts disabled, so the guest
/// takes none while it runs.
const QUIET_RFLAGS: u64 = 0x2;
/// `DR7`'s reset value.
const DR7_RESET: u64 = 0x400;

/// Programs a flat 64-bit guest that begins at `rip` with stack `rsp`, running
/// in the host's address space.
///
/// # Errors
///
/// The [`VmFail`] from the first `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation, and the
/// controls and host state must be programmed around this. `rip` must be a
/// host-mapped, host-executable linear address, because the guest shares the
/// host's translation; `rsp` must be host-mapped writable memory.
pub unsafe fn program(cell: &Vmcs, rip: u64, rsp: u64) -> Result<(), VmFail> {
    // SAFETY: `cell` is the current VMCS; the control registers, EFER, tables
    // and task register read here are this processor's own, which is what makes
    // the guest's address space the host's.
    unsafe {
        let cr0 = Cr0::read_raw();
        let cr4 = Cr4::read_raw();
        cell.write(Field::GUEST_CR0, cr0)?;
        cell.write(Field::GUEST_CR3, host::read_cr3())?;
        cell.write(Field::GUEST_CR4, cr4)?;
        cell.write(Field::GUEST_IA32_EFER, msr::rdmsr(IA32_EFER))?;
        cell.write(Field::GUEST_DR7, DR7_RESET)?;
        cell.write(Field::GUEST_RFLAGS, QUIET_RFLAGS)?;
        cell.write(Field::GUEST_RIP, rip)?;
        cell.write(Field::GUEST_RSP, rsp)?;

        // The read shadows the guest sees for its control registers; with the
        // masks empty they are not consulted, but a valid VMCS carries them.
        cell.write(Field::CR0_READ_SHADOW, cr0)?;
        cell.write(Field::CR4_READ_SHADOW, cr4)?;

        write_segment(cell, Segment::Cs, CODE_SELECTOR, CODE_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Ss, DATA_SELECTOR, DATA_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Ds, DATA_SELECTOR, DATA_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Es, DATA_SELECTOR, DATA_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Fs, DATA_SELECTOR, DATA_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Gs, DATA_SELECTOR, DATA_ACCESS_RIGHTS)?;
        write_segment(cell, Segment::Ldtr, 0, UNUSABLE_ACCESS_RIGHTS)?;

        // The task register is the host's, so the guest has a valid one without
        // a GDT of its own; its base is the host TSS's.
        let tr = host::read_tr();
        let gdt = sgdt();
        let idt = sidt();
        cell.write(Field::GUEST_TR_SELECTOR, u64::from(tr))?;
        cell.write(Field::GUEST_TR_BASE, host::tss_base(gdt.base, tr))?;
        cell.write(Field::GUEST_TR_LIMIT, TASK_LIMIT)?;
        cell.write(Field::GUEST_TR_ACCESS_RIGHTS, TASK_ACCESS_RIGHTS)?;

        cell.write(Field::GUEST_GDTR_BASE, gdt.base.as_u64())?;
        cell.write(Field::GUEST_GDTR_LIMIT, u64::from(gdt.limit))?;
        cell.write(Field::GUEST_IDTR_BASE, idt.base.as_u64())?;
        cell.write(Field::GUEST_IDTR_LIMIT, u64::from(idt.limit))?;

        cell.write(Field::GUEST_IA32_SYSENTER_CS, 0)?;
        cell.write(Field::GUEST_IA32_SYSENTER_ESP, 0)?;
        cell.write(Field::GUEST_IA32_SYSENTER_EIP, 0)?;
        cell.write(Field::GUEST_INTERRUPTIBILITY_STATE, 0)?;
        cell.write(Field::GUEST_ACTIVITY_STATE, 0)?;
        cell.write(Field::GUEST_PENDING_DBG_EXCEPTIONS, 0)?;
    }
    Ok(())
}

/// Which segment register a set of fields describes.
#[derive(Clone, Copy)]
enum Segment {
    Cs,
    Ss,
    Ds,
    Es,
    Fs,
    Gs,
    Ldtr,
}

impl Segment {
    /// The four VMCS fields that describe this segment: selector, base, limit
    /// and access rights.
    const fn fields(
        self,
    ) -> (
        vmx::FieldEncoding,
        vmx::FieldEncoding,
        vmx::FieldEncoding,
        vmx::FieldEncoding,
    ) {
        match self {
            Self::Cs => (
                Field::GUEST_CS_SELECTOR,
                Field::GUEST_CS_BASE,
                Field::GUEST_CS_LIMIT,
                Field::GUEST_CS_ACCESS_RIGHTS,
            ),
            Self::Ss => (
                Field::GUEST_SS_SELECTOR,
                Field::GUEST_SS_BASE,
                Field::GUEST_SS_LIMIT,
                Field::GUEST_SS_ACCESS_RIGHTS,
            ),
            Self::Ds => (
                Field::GUEST_DS_SELECTOR,
                Field::GUEST_DS_BASE,
                Field::GUEST_DS_LIMIT,
                Field::GUEST_DS_ACCESS_RIGHTS,
            ),
            Self::Es => (
                Field::GUEST_ES_SELECTOR,
                Field::GUEST_ES_BASE,
                Field::GUEST_ES_LIMIT,
                Field::GUEST_ES_ACCESS_RIGHTS,
            ),
            Self::Fs => (
                Field::GUEST_FS_SELECTOR,
                Field::GUEST_FS_BASE,
                Field::GUEST_FS_LIMIT,
                Field::GUEST_FS_ACCESS_RIGHTS,
            ),
            Self::Gs => (
                Field::GUEST_GS_SELECTOR,
                Field::GUEST_GS_BASE,
                Field::GUEST_GS_LIMIT,
                Field::GUEST_GS_ACCESS_RIGHTS,
            ),
            Self::Ldtr => (
                Field::GUEST_LDTR_SELECTOR,
                Field::GUEST_LDTR_BASE,
                Field::GUEST_LDTR_LIMIT,
                Field::GUEST_LDTR_ACCESS_RIGHTS,
            ),
        }
    }
}

/// Writes a flat segment: base zero, limit the whole space, with `selector` and
/// `access_rights`. An unusable segment still takes a flat limit, which the
/// entry ignores.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor.
unsafe fn write_segment(
    cell: &Vmcs,
    segment: Segment,
    selector: u64,
    access_rights: u64,
) -> Result<(), VmFail> {
    let (selector_field, base_field, limit_field, rights_field) = segment.fields();
    // SAFETY: the caller guarantees the current VMCS.
    unsafe {
        cell.write(selector_field, selector)?;
        cell.write(base_field, 0)?;
        cell.write(limit_field, FLAT_LIMIT)?;
        cell.write(rights_field, access_rights)?;
    }
    Ok(())
}
