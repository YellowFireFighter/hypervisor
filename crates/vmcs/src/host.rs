//! Writing the host state a VM exit restores.
//!
//! When the guest exits, the processor loads the host's control registers,
//! segment selectors and bases, descriptor-table bases and a few model-specific
//! registers from the VMCS — everything except the general registers, which the
//! run loop saves and restores itself, and the host `RIP` and `RSP`, which the
//! run loop writes just before entry. So this captures the processor the host
//! is running on right now and writes it into the VMCS, which is what makes a
//! VM exit return into a working hypervisor rather than a corrupted one.
//!
//! The host selectors must have a zero requestor-privilege and table-indicator,
//! which the running host's already do; they are masked to the index to be
//! certain. The one awkward field is the task-register base, which is not a
//! register but a descriptor in the GDT, so it is read from there.

use vmx::Field;
use x86_64::{
    VirtAddr,
    instructions::tables::{sgdt, sidt},
    registers::{
        control::{Cr0, Cr4},
        segmentation::{CS, DS, ES, FS, GS, SS, Segment},
    },
};

use crate::{
    Vmcs,
    error::VmFail,
    msr::{self, IA32_EFER, IA32_FS_BASE, IA32_GS_BASE},
};

/// Bits of a selector below the index: the requestor-privilege level and the
/// table indicator, both of which a host selector must clear.
const SELECTOR_INDEX_MASK: u64 = !0b111;

/// Writes the running host's state into the current VMCS.
///
/// Host `RIP` and `RSP` are not written here — the run loop sets them to its
/// own return label and stack immediately before each entry.
///
/// # Errors
///
/// The [`VmFail`] from the first `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, and this must run on the
/// processor whose state is to be restored on exit — the state read here is
/// this processor's own.
pub unsafe fn program(cell: &Vmcs) -> Result<(), VmFail> {
    // SAFETY: `cell` is the current VMCS; every value below is read from this
    // processor's own registers and tables.
    unsafe {
        cell.write(Field::HOST_CR0, Cr0::read_raw())?;
        cell.write(Field::HOST_CR3, read_cr3())?;
        cell.write(Field::HOST_CR4, Cr4::read_raw())?;

        cell.write(Field::HOST_CS_SELECTOR, selector(CS::get_reg().0))?;
        cell.write(Field::HOST_SS_SELECTOR, selector(SS::get_reg().0))?;
        cell.write(Field::HOST_DS_SELECTOR, selector(DS::get_reg().0))?;
        cell.write(Field::HOST_ES_SELECTOR, selector(ES::get_reg().0))?;
        cell.write(Field::HOST_FS_SELECTOR, selector(FS::get_reg().0))?;
        cell.write(Field::HOST_GS_SELECTOR, selector(GS::get_reg().0))?;

        let tr = read_tr();
        let gdt = sgdt();
        let idt = sidt();
        cell.write(Field::HOST_TR_SELECTOR, selector(tr))?;
        cell.write(Field::HOST_TR_BASE, tss_base(gdt.base, tr))?;
        cell.write(Field::HOST_GDTR_BASE, gdt.base.as_u64())?;
        cell.write(Field::HOST_IDTR_BASE, idt.base.as_u64())?;

        cell.write(Field::HOST_FS_BASE, msr::rdmsr(IA32_FS_BASE))?;
        cell.write(Field::HOST_GS_BASE, msr::rdmsr(IA32_GS_BASE))?;
        cell.write(Field::HOST_IA32_EFER, msr::rdmsr(IA32_EFER))?;

        // The host enters its interrupt handlers through its own descriptor
        // tables, not through the fast-system-call registers, so these are left
        // at zero rather than mirrored — a host that does not use SYSENTER has
        // nothing to restore for it.
        cell.write(Field::HOST_IA32_SYSENTER_CS, 0)?;
        cell.write(Field::HOST_IA32_SYSENTER_ESP, 0)?;
        cell.write(Field::HOST_IA32_SYSENTER_EIP, 0)?;
    }
    Ok(())
}

/// A selector with its privilege and table bits cleared, widened for the field.
fn selector(raw: u16) -> u64 {
    u64::from(raw) & SELECTOR_INDEX_MASK
}

/// Reads `CR3`.
///
/// # Safety
///
/// Must run in a context allowed to read control registers, which host code in
/// long mode is.
pub(crate) unsafe fn read_cr3() -> u64 {
    let value: u64;
    // SAFETY: reading CR3 has no precondition beyond the privilege host code
    // holds; it touches no memory and sets no flags.
    unsafe {
        core::arch::asm!("mov {}, cr3", out(reg) value, options(nomem, nostack, preserves_flags));
    }
    value
}

/// Reads the task-register selector.
///
/// # Safety
///
/// Must run where `STR` is permitted, which host code is.
pub(crate) unsafe fn read_tr() -> u16 {
    let tr: u16;
    // SAFETY: `STR` stores the visible task register and has no precondition in
    // host code; it touches no memory and sets no flags.
    unsafe {
        core::arch::asm!("str {0:x}", out(reg) tr, options(nomem, nostack, preserves_flags));
    }
    tr
}

/// The base address a task-state-segment descriptor in the GDT holds.
///
/// The 64-bit system descriptor spreads the base across four runs; this
/// reassembles them. `selector` is the task register, whose index names the
/// descriptor.
///
/// # Safety
///
/// `gdt_base` must be the live GDT's base and `selector` an in-range task
/// register, so the sixteen bytes read are a real descriptor.
pub(crate) unsafe fn tss_base(gdt_base: VirtAddr, selector: u16) -> u64 {
    let descriptor =
        (gdt_base.as_u64() + (u64::from(selector) & SELECTOR_INDEX_MASK)) as *const u32;
    // SAFETY: the caller guarantees `descriptor` points at a 16-byte system
    // descriptor in the live GDT; the three reads stay within it.
    let (low, mid, high) = unsafe {
        (
            descriptor.read(),
            descriptor.add(1).read(),
            descriptor.add(2).read(),
        )
    };
    let base_15_0 = u64::from(low >> 16);
    let base_23_16 = u64::from(mid & 0xFF);
    let base_31_24 = u64::from(mid >> 24);
    let base_63_32 = u64::from(high);
    base_15_0 | (base_23_16 << 16) | (base_31_24 << 24) | (base_63_32 << 32)
}
