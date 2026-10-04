//! Reading the guest state back out of the current VMCS, for checking.
//!
//! A refused VM entry leaves the state it refused in the VMCS, and
//! [`vmx::check`] can say which rule it broke — but only from values, and
//! those have to come out of the VMCS one `VMREAD` at a time, together with the
//! processor's own limits the rules are measured against. This gathers both.
//! Reading back rather than reusing what was written is the point: it is the
//! VMCS the processor checks, so it is the VMCS that is checked here.
//!
//! `VMREAD` zero-extends a 16- or 32-bit field into the full register, so each
//! narrowing below loses nothing; a value that somehow did not fit reads as all
//! ones, which the checks treat as broken rather than as plausible.

use core::arch::x86_64::__cpuid;

use vmx::{
    Field, FieldEncoding, VmEntry,
    check::{GuestSegment, GuestState, GuestTable, Limits},
};

use crate::{
    Vmcs,
    error::VmFail,
    fixed::{IA32_VMX_CR0_FIXED0, IA32_VMX_CR0_FIXED1, IA32_VMX_CR4_FIXED0, IA32_VMX_CR4_FIXED1},
    msr,
};

/// The extended `CPUID` leaf that reports the address widths.
const ADDRESS_WIDTHS_LEAF: u32 = 0x8000_0008;
/// Bits the linear-address width sits above the physical one in that leaf's
/// `EAX`.
const LINEAR_WIDTH_SHIFT: u32 = 8;

/// Reads the guest-state area, and the entry controls its checks depend on, out
/// of the current VMCS.
///
/// # Errors
///
/// The [`VmFail`] from the first `VMREAD` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn guest_state(cell: &Vmcs) -> Result<GuestState, VmFail> {
    // SAFETY: the caller guarantees the current VMCS; every field read here is a
    // guest-state or control field every VMCS has.
    unsafe {
        let entry = u32::try_from(cell.read(Field::VM_ENTRY_CONTROLS)?).unwrap_or(u32::MAX);
        Ok(GuestState {
            entry: VmEntry::from_bits_retain(entry),
            cr0: cell.read(Field::GUEST_CR0)?,
            cr3: cell.read(Field::GUEST_CR3)?,
            cr4: cell.read(Field::GUEST_CR4)?,
            dr7: cell.read(Field::GUEST_DR7)?,
            efer: cell.read(Field::GUEST_IA32_EFER)?,
            rflags: cell.read(Field::GUEST_RFLAGS)?,
            rip: cell.read(Field::GUEST_RIP)?,
            sysenter_esp: cell.read(Field::GUEST_IA32_SYSENTER_ESP)?,
            sysenter_eip: cell.read(Field::GUEST_IA32_SYSENTER_EIP)?,
            es: segment(
                cell,
                Field::GUEST_ES_SELECTOR,
                Field::GUEST_ES_BASE,
                Field::GUEST_ES_LIMIT,
                Field::GUEST_ES_ACCESS_RIGHTS,
            )?,
            cs: segment(
                cell,
                Field::GUEST_CS_SELECTOR,
                Field::GUEST_CS_BASE,
                Field::GUEST_CS_LIMIT,
                Field::GUEST_CS_ACCESS_RIGHTS,
            )?,
            ss: segment(
                cell,
                Field::GUEST_SS_SELECTOR,
                Field::GUEST_SS_BASE,
                Field::GUEST_SS_LIMIT,
                Field::GUEST_SS_ACCESS_RIGHTS,
            )?,
            ds: segment(
                cell,
                Field::GUEST_DS_SELECTOR,
                Field::GUEST_DS_BASE,
                Field::GUEST_DS_LIMIT,
                Field::GUEST_DS_ACCESS_RIGHTS,
            )?,
            fs: segment(
                cell,
                Field::GUEST_FS_SELECTOR,
                Field::GUEST_FS_BASE,
                Field::GUEST_FS_LIMIT,
                Field::GUEST_FS_ACCESS_RIGHTS,
            )?,
            gs: segment(
                cell,
                Field::GUEST_GS_SELECTOR,
                Field::GUEST_GS_BASE,
                Field::GUEST_GS_LIMIT,
                Field::GUEST_GS_ACCESS_RIGHTS,
            )?,
            ldtr: segment(
                cell,
                Field::GUEST_LDTR_SELECTOR,
                Field::GUEST_LDTR_BASE,
                Field::GUEST_LDTR_LIMIT,
                Field::GUEST_LDTR_ACCESS_RIGHTS,
            )?,
            tr: segment(
                cell,
                Field::GUEST_TR_SELECTOR,
                Field::GUEST_TR_BASE,
                Field::GUEST_TR_LIMIT,
                Field::GUEST_TR_ACCESS_RIGHTS,
            )?,
            gdtr: table(cell, Field::GUEST_GDTR_BASE, Field::GUEST_GDTR_LIMIT)?,
            idtr: table(cell, Field::GUEST_IDTR_BASE, Field::GUEST_IDTR_LIMIT)?,
        })
    }
}

/// The processor limits the guest-state checks are measured against: the
/// control-register fixed bits and the address widths.
///
/// # Safety
///
/// This processor must support VMX, so the fixed-bit registers exist.
#[must_use]
pub unsafe fn limits() -> Limits {
    let widths = __cpuid(ADDRESS_WIDTHS_LEAF).eax;
    // SAFETY: the caller guarantees VMX support, on which the four fixed-bit
    // registers exist.
    unsafe {
        Limits {
            cr0_fixed0: msr::rdmsr(IA32_VMX_CR0_FIXED0),
            cr0_fixed1: msr::rdmsr(IA32_VMX_CR0_FIXED1),
            cr4_fixed0: msr::rdmsr(IA32_VMX_CR4_FIXED0),
            cr4_fixed1: msr::rdmsr(IA32_VMX_CR4_FIXED1),
            physical_address_bits: widths.to_le_bytes()[0],
            linear_address_bits: (widths >> LINEAR_WIDTH_SHIFT).to_le_bytes()[0],
        }
    }
}

/// Reads one guest segment's four fields.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor.
unsafe fn segment(
    cell: &Vmcs,
    selector: FieldEncoding,
    base: FieldEncoding,
    limit: FieldEncoding,
    rights: FieldEncoding,
) -> Result<GuestSegment, VmFail> {
    // SAFETY: the caller guarantees the current VMCS.
    unsafe {
        Ok(GuestSegment {
            selector: u16::try_from(cell.read(selector)?).unwrap_or(u16::MAX),
            base: cell.read(base)?,
            limit: u32::try_from(cell.read(limit)?).unwrap_or(u32::MAX),
            rights: u32::try_from(cell.read(rights)?).unwrap_or(u32::MAX),
        })
    }
}

/// Reads one guest descriptor-table register's two fields.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor.
unsafe fn table(
    cell: &Vmcs,
    base: FieldEncoding,
    limit: FieldEncoding,
) -> Result<GuestTable, VmFail> {
    // SAFETY: the caller guarantees the current VMCS.
    unsafe {
        Ok(GuestTable {
            base: cell.read(base)?,
            limit: u32::try_from(cell.read(limit)?).unwrap_or(u32::MAX),
        })
    }
}
