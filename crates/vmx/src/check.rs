//! The guest-state checks a VM entry applies, restated so a refused entry can
//! say which one it failed.
//!
//! When the processor refuses a guest's state it exits with basic reason 33
//! and a zero exit qualification. It does not say which of the several dozen
//! checks on the guest-state area failed, and nothing it reports afterwards
//! narrows that down. So this module restates those checks over values a caller
//! reads back out of the VMCS, and names every one the state breaks.
//!
//! It covers the guest a 64-bit hypervisor enters: protected mode with paging,
//! never virtual-8086 mode, and without the unrestricted-guest control, which
//! would relax the checks on `CR0` and on segment privilege. It leaves out the
//! state the callers in this workspace never load or always load as zero — the
//! performance-counter, PAT and control-flow-enforcement registers, the debug
//! control register, the activity and interruptibility state, the pending debug
//! exceptions, and the VMCS link pointer, which the processor reports through
//! its own exit qualification anyway.
//!
//! So a violation reported here is one of the processor's own rules restated,
//! and an entry with that state is certain to fail. A clean report is weaker:
//! the entry can still fail on one of the checks this leaves out.

use core::fmt::{self, Display, Formatter};

use x86_64::registers::{
    control::{Cr0Flags, Cr4Flags},
    model_specific::EferFlags,
    rflags::RFlags,
};

use crate::{
    AccessRights, VmEntry,
    segment::{
        BUSY_TSS_16_TYPE, BUSY_TSS_TYPE, LDT_TYPE, TYPE_ACCESSED, TYPE_CODE, TYPE_CONFORMING,
        TYPE_READ_WRITE,
    },
};

/// The `RFLAGS` bit that always reads as one, and must be one on entry.
const RFLAGS_FIXED_ONE: u64 = 1 << 1;

/// The bits of an access-rights word the architecture reserves: 11:8 and
/// 31:17.
const RIGHTS_RESERVED: u32 = 0xFFFE_0F00;

/// The low limit bits a segment with page granularity must have all set.
const LIMIT_PAGE_OFFSET: u32 = 0xFFF;
/// The high limit bits a segment with byte granularity must have all clear.
const LIMIT_ABOVE_BYTE_GRANULAR: u32 = 0xFFF0_0000;

/// The bits a descriptor-table limit field may hold.
const TABLE_LIMIT_MASK: u32 = 0xFFFF;

/// The selector bit that names the local descriptor table.
const SELECTOR_LOCAL_TABLE: u16 = 1 << 2;
/// The selector bits that hold the requested privilege level.
const SELECTOR_PRIVILEGE: u16 = 0b11;

/// The linear-address width a processor without five-level paging has, which
/// is what `CR4.LA57` clear leaves a 64-bit guest with.
const FOUR_LEVEL_ADDRESS_BITS: u8 = 48;

/// A guest segment as the VMCS holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestSegment {
    /// The selector field.
    pub selector: u16,
    /// The base-address field.
    pub base: u64,
    /// The limit field, in bytes.
    pub limit: u32,
    /// The access-rights field, raw, so reserved bits the processor would
    /// reject are seen rather than masked away.
    pub rights: u32,
}

impl GuestSegment {
    /// The access-rights word, decoded.
    fn decoded(&self) -> AccessRights {
        AccessRights::from_bits(self.rights)
    }

    /// The privilege level the selector requests.
    fn requested_privilege(&self) -> u8 {
        // Masked to the field's own two bits, so the narrowing loses nothing.
        (self.selector & SELECTOR_PRIVILEGE) as u8
    }
}

/// A descriptor-table register as the VMCS holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestTable {
    /// The base-address field.
    pub base: u64,
    /// The limit field.
    pub limit: u32,
}

/// The guest-state area of a VMCS, and the entry controls its checks depend
/// on.
#[derive(Clone, Copy, Debug)]
pub struct GuestState {
    /// The VM-entry controls, which say whether the guest enters 64-bit mode
    /// and which registers the entry loads.
    pub entry: VmEntry,
    /// `CR0`.
    pub cr0: u64,
    /// `CR3`.
    pub cr3: u64,
    /// `CR4`.
    pub cr4: u64,
    /// `DR7`.
    pub dr7: u64,
    /// `IA32_EFER`.
    pub efer: u64,
    /// `RFLAGS`.
    pub rflags: u64,
    /// `RIP`.
    pub rip: u64,
    /// `IA32_SYSENTER_ESP`.
    pub sysenter_esp: u64,
    /// `IA32_SYSENTER_EIP`.
    pub sysenter_eip: u64,
    /// `ES`.
    pub es: GuestSegment,
    /// `CS`.
    pub cs: GuestSegment,
    /// `SS`.
    pub ss: GuestSegment,
    /// `DS`.
    pub ds: GuestSegment,
    /// `FS`.
    pub fs: GuestSegment,
    /// `GS`.
    pub gs: GuestSegment,
    /// The local descriptor table register.
    pub ldtr: GuestSegment,
    /// The task register.
    pub tr: GuestSegment,
    /// The global descriptor table register.
    pub gdtr: GuestTable,
    /// The interrupt descriptor table register.
    pub idtr: GuestTable,
}

/// What the processor allows, which several checks are measured against.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// `IA32_VMX_CR0_FIXED0`: the `CR0` bits that must be set.
    pub cr0_fixed0: u64,
    /// `IA32_VMX_CR0_FIXED1`: the `CR0` bits that may be set.
    pub cr0_fixed1: u64,
    /// `IA32_VMX_CR4_FIXED0`: the `CR4` bits that must be set.
    pub cr4_fixed0: u64,
    /// `IA32_VMX_CR4_FIXED1`: the `CR4` bits that may be set.
    pub cr4_fixed1: u64,
    /// The processor's physical-address width, in bits.
    pub physical_address_bits: u8,
    /// The processor's linear-address width, in bits.
    pub linear_address_bits: u8,
}

/// Which part of the guest state a violation is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Subject {
    /// `CR0`.
    Cr0,
    /// `CR3`.
    Cr3,
    /// `CR4`.
    Cr4,
    /// `DR7`.
    Dr7,
    /// `IA32_EFER`.
    Efer,
    /// `RFLAGS`.
    Rflags,
    /// `RIP`.
    Rip,
    /// `IA32_SYSENTER_ESP`.
    SysenterEsp,
    /// `IA32_SYSENTER_EIP`.
    SysenterEip,
    /// The global descriptor table register.
    Gdtr,
    /// The interrupt descriptor table register.
    Idtr,
    /// `ES`.
    Es,
    /// `CS`.
    Cs,
    /// `SS`.
    Ss,
    /// `DS`.
    Ds,
    /// `FS`.
    Fs,
    /// `GS`.
    Gs,
    /// The local descriptor table register.
    Ldtr,
    /// The task register.
    Tr,
}

impl Display for Subject {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cr0 => "cr0",
            Self::Cr3 => "cr3",
            Self::Cr4 => "cr4",
            Self::Dr7 => "dr7",
            Self::Efer => "efer",
            Self::Rflags => "rflags",
            Self::Rip => "rip",
            Self::SysenterEsp => "sysenter_esp",
            Self::SysenterEip => "sysenter_eip",
            Self::Gdtr => "gdtr",
            Self::Idtr => "idtr",
            Self::Es => "es",
            Self::Cs => "cs",
            Self::Ss => "ss",
            Self::Ds => "ds",
            Self::Fs => "fs",
            Self::Gs => "gs",
            Self::Ldtr => "ldtr",
            Self::Tr => "tr",
        })
    }
}

/// Which rule a violation breaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    /// A bit the fixed-bit registers force on is clear, or one they forbid is
    /// set.
    FixedBits,
    /// Paging is on without protected mode.
    PagingWithoutProtection,
    /// Control-flow enforcement is on without write protection.
    EnforcementWithoutWriteProtect,
    /// The guest enters 64-bit mode without paging or physical-address
    /// extension.
    LongModeWithoutPaging,
    /// Process-context identifiers are on outside 64-bit mode.
    ContextIdsOutsideLongMode,
    /// A bit is set above the processor's physical-address width.
    BeyondPhysicalWidth,
    /// A bit is set in the upper half, which must be zero.
    UpperHalfSet,
    /// A reserved bit is set, or a bit that must be one is clear.
    Reserved,
    /// `EFER`'s long-mode bits disagree with the 64-bit-guest entry control.
    LongModeMismatch,
    /// Virtual-8086 mode is on.
    Virtual8086,
    /// The address is not canonical.
    NotCanonical,
    /// The selector names the local descriptor table.
    LocalTableSelector,
    /// The stack selector requests a different privilege than the code
    /// selector does.
    RequestedPrivilege,
    /// The descriptor privilege breaks its rule against the selector or the
    /// code segment.
    DescriptorPrivilege,
    /// The segment is marked unusable, which this register may not be.
    Unusable,
    /// The segment's type is not one this register may hold.
    Type,
    /// The descriptor bit is wrong: a system segment where code or data
    /// belongs, or the reverse.
    DescriptorKind,
    /// The segment is not present.
    NotPresent,
    /// A 64-bit code segment also claims the 32-bit default size.
    LongWithDefaultSize,
    /// The limit and the granularity bit disagree.
    Granularity,
    /// A descriptor-table limit is wider than sixteen bits.
    TableLimit,
}

impl Display for Rule {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::FixedBits => "breaks the VMX fixed bits",
            Self::PagingWithoutProtection => "paging without protected mode",
            Self::EnforcementWithoutWriteProtect => "CET without CR0.WP",
            Self::LongModeWithoutPaging => "64-bit guest without CR0.PG and CR4.PAE",
            Self::ContextIdsOutsideLongMode => "PCIDE outside 64-bit mode",
            Self::BeyondPhysicalWidth => "bits above the physical-address width",
            Self::UpperHalfSet => "upper 32 bits must be zero",
            Self::Reserved => "reserved bits wrong",
            Self::LongModeMismatch => "LMA/LME disagree with the 64-bit-guest control",
            Self::Virtual8086 => "virtual-8086 mode",
            Self::NotCanonical => "not canonical",
            Self::LocalTableSelector => "selector names the LDT",
            Self::RequestedPrivilege => "SS.RPL differs from CS.RPL",
            Self::DescriptorPrivilege => "DPL breaks its rule",
            Self::Unusable => "unusable, which this register may not be",
            Self::Type => "wrong segment type",
            Self::DescriptorKind => "wrong descriptor (S) bit",
            Self::NotPresent => "not present",
            Self::LongWithDefaultSize => "both L and D/B set",
            Self::Granularity => "limit disagrees with G",
            Self::TableLimit => "limit wider than 16 bits",
        })
    }
}

/// One guest-state check the state fails: where, which rule, and the value
/// that breaks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Violation {
    /// The part of the guest state.
    pub subject: Subject,
    /// The rule it breaks.
    pub rule: Rule,
    /// The offending value: the register, the address, or, for a segment's
    /// type, privilege, presence and granularity rules, its access rights.
    pub value: u64,
}

impl Display for Violation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} {:#x}: {}",
            self.subject, self.value, self.rule
        )
    }
}

/// Checks `state` against the guest-state rules a VM entry applies, calling
/// `report` once for each rule it breaks, in the order the architecture lists
/// them: control registers and model-specific registers, then segments, then
/// descriptor tables, then `RIP` and `RFLAGS`.
pub fn check(state: &GuestState, limits: &Limits, mut report: impl FnMut(Violation)) {
    let long_mode = state.entry.contains(VmEntry::IA32E_MODE_GUEST);
    let mut found = |subject: Subject, rule: Rule, value: u64| {
        report(Violation {
            subject,
            rule,
            value,
        });
    };
    registers(state, limits, long_mode, &mut found);
    segments(state, limits, long_mode, &mut found);
    tables(state, limits, &mut found);
    instruction_and_flags(state, limits, long_mode, &mut found);
}

/// The checks on the control, debug and model-specific registers.
fn registers(
    state: &GuestState,
    limits: &Limits,
    long_mode: bool,
    found: &mut impl FnMut(Subject, Rule, u64),
) {
    let cr0 = Cr0Flags::from_bits_retain(state.cr0);
    let cr4 = Cr4Flags::from_bits_retain(state.cr4);
    if !fixed(state.cr0, limits.cr0_fixed0, limits.cr0_fixed1) {
        found(Subject::Cr0, Rule::FixedBits, state.cr0);
    }
    if cr0.contains(Cr0Flags::PAGING) && !cr0.contains(Cr0Flags::PROTECTED_MODE_ENABLE) {
        found(Subject::Cr0, Rule::PagingWithoutProtection, state.cr0);
    }
    if !fixed(state.cr4, limits.cr4_fixed0, limits.cr4_fixed1) {
        found(Subject::Cr4, Rule::FixedBits, state.cr4);
    }
    if cr4.contains(Cr4Flags::CONTROL_FLOW_ENFORCEMENT) && !cr0.contains(Cr0Flags::WRITE_PROTECT) {
        found(
            Subject::Cr4,
            Rule::EnforcementWithoutWriteProtect,
            state.cr4,
        );
    }
    if state.entry.contains(VmEntry::LOAD_DEBUG_CONTROLS) && upper_half(state.dr7) {
        found(Subject::Dr7, Rule::UpperHalfSet, state.dr7);
    }
    if long_mode {
        if !cr0.contains(Cr0Flags::PAGING) || !cr4.contains(Cr4Flags::PHYSICAL_ADDRESS_EXTENSION) {
            found(Subject::Cr4, Rule::LongModeWithoutPaging, state.cr4);
        }
    } else if cr4.contains(Cr4Flags::PCID) {
        found(Subject::Cr4, Rule::ContextIdsOutsideLongMode, state.cr4);
    }
    if state.cr3 >> limits.physical_address_bits != 0 {
        found(Subject::Cr3, Rule::BeyondPhysicalWidth, state.cr3);
    }
    let width = linear_width(state, limits);
    if !canonical(state.sysenter_esp, width) {
        found(Subject::SysenterEsp, Rule::NotCanonical, state.sysenter_esp);
    }
    if !canonical(state.sysenter_eip, width) {
        found(Subject::SysenterEip, Rule::NotCanonical, state.sysenter_eip);
    }
    if state.entry.contains(VmEntry::LOAD_IA32_EFER) {
        efer(state, cr0, long_mode, found);
    }
}

/// The checks on `IA32_EFER`, which apply only when the entry loads it.
fn efer(
    state: &GuestState,
    cr0: Cr0Flags,
    long_mode: bool,
    found: &mut impl FnMut(Subject, Rule, u64),
) {
    let defined = EferFlags::SYSTEM_CALL_EXTENSIONS
        | EferFlags::LONG_MODE_ENABLE
        | EferFlags::LONG_MODE_ACTIVE
        | EferFlags::NO_EXECUTE_ENABLE;
    let efer = EferFlags::from_bits_retain(state.efer);
    if state.efer & !defined.bits() != 0 {
        found(Subject::Efer, Rule::Reserved, state.efer);
    }
    let active = efer.contains(EferFlags::LONG_MODE_ACTIVE);
    let enabled = efer.contains(EferFlags::LONG_MODE_ENABLE);
    if active != long_mode || (cr0.contains(Cr0Flags::PAGING) && enabled != active) {
        found(Subject::Efer, Rule::LongModeMismatch, state.efer);
    }
}

/// The checks on the eight segment registers.
fn segments(
    state: &GuestState,
    limits: &Limits,
    long_mode: bool,
    found: &mut impl FnMut(Subject, Rule, u64),
) {
    let width = linear_width(state, limits);
    if state.tr.selector & SELECTOR_LOCAL_TABLE != 0 {
        found(
            Subject::Tr,
            Rule::LocalTableSelector,
            u64::from(state.tr.selector),
        );
    }
    if state.ss.requested_privilege() != state.cs.requested_privilege() {
        found(
            Subject::Ss,
            Rule::RequestedPrivilege,
            u64::from(state.ss.selector),
        );
    }
    for (subject, segment) in [
        (Subject::Tr, &state.tr),
        (Subject::Fs, &state.fs),
        (Subject::Gs, &state.gs),
    ] {
        if !canonical(segment.base, width) {
            found(subject, Rule::NotCanonical, segment.base);
        }
    }
    if upper_half(state.cs.base) {
        found(Subject::Cs, Rule::UpperHalfSet, state.cs.base);
    }
    for (subject, segment) in [
        (Subject::Ss, &state.ss),
        (Subject::Ds, &state.ds),
        (Subject::Es, &state.es),
    ] {
        if usable(segment) && upper_half(segment.base) {
            found(subject, Rule::UpperHalfSet, segment.base);
        }
    }

    code(state, long_mode, found);
    stack(state, found);
    for (subject, segment) in [
        (Subject::Ds, &state.ds),
        (Subject::Es, &state.es),
        (Subject::Fs, &state.fs),
        (Subject::Gs, &state.gs),
    ] {
        data(subject, segment, found);
    }
    task(state, long_mode, found);
    local(state, width, found);
}

/// The checks on `CS`, which must always be usable.
fn code(state: &GuestState, long_mode: bool, found: &mut impl FnMut(Subject, Rule, u64)) {
    let cs = &state.cs;
    let rights = cs.decoded();
    let raw = u64::from(cs.rights);
    if rights.unusable() {
        found(Subject::Cs, Rule::Unusable, raw);
        return;
    }
    let kind = rights.kind();
    if kind & (TYPE_CODE | TYPE_ACCESSED) != TYPE_CODE | TYPE_ACCESSED {
        found(Subject::Cs, Rule::Type, raw);
    }
    let ss_dpl = state.ss.decoded().dpl();
    let privilege_holds = if kind & TYPE_CONFORMING == 0 {
        rights.dpl() == ss_dpl
    } else {
        rights.dpl() <= ss_dpl
    };
    if !privilege_holds {
        found(Subject::Cs, Rule::DescriptorPrivilege, raw);
    }
    if long_mode && rights.long() && rights.default_size() {
        found(Subject::Cs, Rule::LongWithDefaultSize, raw);
    }
    common(Subject::Cs, cs, true, found);
}

/// The checks on `SS`, when it is usable.
fn stack(state: &GuestState, found: &mut impl FnMut(Subject, Rule, u64)) {
    let ss = &state.ss;
    if !usable(ss) {
        return;
    }
    let rights = ss.decoded();
    let raw = u64::from(ss.rights);
    let kind = rights.kind();
    if kind & !TYPE_CONFORMING != TYPE_READ_WRITE | TYPE_ACCESSED {
        found(Subject::Ss, Rule::Type, raw);
    }
    if rights.dpl() != ss.requested_privilege() {
        found(Subject::Ss, Rule::DescriptorPrivilege, raw);
    }
    common(Subject::Ss, ss, true, found);
}

/// The checks on `DS`, `ES`, `FS` or `GS`, when it is usable.
fn data(subject: Subject, segment: &GuestSegment, found: &mut impl FnMut(Subject, Rule, u64)) {
    if !usable(segment) {
        return;
    }
    let rights = segment.decoded();
    let raw = u64::from(segment.rights);
    let kind = rights.kind();
    let unreadable_code = kind & TYPE_CODE != 0 && kind & TYPE_READ_WRITE == 0;
    if kind & TYPE_ACCESSED == 0 || unreadable_code {
        found(subject, Rule::Type, raw);
    }
    // Data and non-conforming code may not be described as more privileged
    // than the selector that loaded them.
    let conforming_code = kind & (TYPE_CODE | TYPE_CONFORMING) == TYPE_CODE | TYPE_CONFORMING;
    if !conforming_code && rights.dpl() < segment.requested_privilege() {
        found(subject, Rule::DescriptorPrivilege, raw);
    }
    common(subject, segment, true, found);
}

/// The checks on the task register, which must always be usable.
fn task(state: &GuestState, long_mode: bool, found: &mut impl FnMut(Subject, Rule, u64)) {
    let tr = &state.tr;
    let rights = tr.decoded();
    let raw = u64::from(tr.rights);
    if rights.unusable() {
        found(Subject::Tr, Rule::Unusable, raw);
        return;
    }
    let kind = rights.kind();
    let allowed = kind == BUSY_TSS_TYPE || (!long_mode && kind == BUSY_TSS_16_TYPE);
    if !allowed {
        found(Subject::Tr, Rule::Type, raw);
    }
    common(Subject::Tr, tr, false, found);
}

/// The checks on the local descriptor table register, when it is usable.
fn local(state: &GuestState, width: u8, found: &mut impl FnMut(Subject, Rule, u64)) {
    let ldtr = &state.ldtr;
    if !usable(ldtr) {
        return;
    }
    if ldtr.selector & SELECTOR_LOCAL_TABLE != 0 {
        found(
            Subject::Ldtr,
            Rule::LocalTableSelector,
            u64::from(ldtr.selector),
        );
    }
    if !canonical(ldtr.base, width) {
        found(Subject::Ldtr, Rule::NotCanonical, ldtr.base);
    }
    if ldtr.decoded().kind() != LDT_TYPE {
        found(Subject::Ldtr, Rule::Type, u64::from(ldtr.rights));
    }
    common(Subject::Ldtr, ldtr, false, found);
}

/// The checks every usable segment shares: the descriptor bit, presence,
/// reserved bits, and the agreement of the limit with the granularity bit.
fn common(
    subject: Subject,
    segment: &GuestSegment,
    code_or_data: bool,
    found: &mut impl FnMut(Subject, Rule, u64),
) {
    let rights = segment.decoded();
    let raw = u64::from(segment.rights);
    if rights.descriptor() != code_or_data {
        found(subject, Rule::DescriptorKind, raw);
    }
    if !rights.present() {
        found(subject, Rule::NotPresent, raw);
    }
    if segment.rights & RIGHTS_RESERVED != 0 {
        found(subject, Rule::Reserved, raw);
    }
    let limit = segment.limit;
    let needs_bytes = limit & LIMIT_PAGE_OFFSET != LIMIT_PAGE_OFFSET;
    let needs_pages = limit & LIMIT_ABOVE_BYTE_GRANULAR != 0;
    if (needs_bytes && rights.granularity()) || (needs_pages && !rights.granularity()) {
        found(subject, Rule::Granularity, raw);
    }
}

/// The checks on the two descriptor-table registers.
fn tables(state: &GuestState, limits: &Limits, found: &mut impl FnMut(Subject, Rule, u64)) {
    let width = linear_width(state, limits);
    for (subject, table) in [(Subject::Gdtr, &state.gdtr), (Subject::Idtr, &state.idtr)] {
        if !canonical(table.base, width) {
            found(subject, Rule::NotCanonical, table.base);
        }
        if table.limit & !TABLE_LIMIT_MASK != 0 {
            found(subject, Rule::TableLimit, u64::from(table.limit));
        }
    }
}

/// The checks on `RIP` and `RFLAGS`.
fn instruction_and_flags(
    state: &GuestState,
    limits: &Limits,
    long_mode: bool,
    found: &mut impl FnMut(Subject, Rule, u64),
) {
    if long_mode && state.cs.decoded().long() {
        if !canonical(state.rip, limits.linear_address_bits) {
            found(Subject::Rip, Rule::NotCanonical, state.rip);
        }
    } else if upper_half(state.rip) {
        found(Subject::Rip, Rule::UpperHalfSet, state.rip);
    }
    let defined = RFlags::all().bits() | RFLAGS_FIXED_ONE;
    if state.rflags & !defined != 0 || state.rflags & RFLAGS_FIXED_ONE == 0 {
        found(Subject::Rflags, Rule::Reserved, state.rflags);
    }
    let cr0 = Cr0Flags::from_bits_retain(state.cr0);
    let virtual_8086 = RFlags::from_bits_retain(state.rflags).contains(RFlags::VIRTUAL_8086_MODE);
    if virtual_8086 && (long_mode || !cr0.contains(Cr0Flags::PROTECTED_MODE_ENABLE)) {
        found(Subject::Rflags, Rule::Virtual8086, state.rflags);
    }
}

/// Whether `value` sets no bit the fixed-bit pair forbids and clears none it
/// forces.
const fn fixed(value: u64, fixed0: u64, fixed1: u64) -> bool {
    value & fixed0 == fixed0 && value & !fixed1 == 0
}

/// Whether a segment is usable, which is when the checks on its contents
/// apply at all.
fn usable(segment: &GuestSegment) -> bool {
    !segment.decoded().unusable()
}

/// Whether any of the upper 32 bits of `value` is set.
const fn upper_half(value: u64) -> bool {
    value >> 32 != 0
}

/// The linear-address width the base-address checks measure against.
///
/// The processor's own width, except that a guest without five-level paging
/// has only the four-level width however wide the processor could go.
fn linear_width(state: &GuestState, limits: &Limits) -> u8 {
    if Cr4Flags::from_bits_retain(state.cr4).contains(Cr4Flags::L5_PAGING) {
        limits.linear_address_bits
    } else {
        limits.linear_address_bits.min(FOUR_LEVEL_ADDRESS_BITS)
    }
}

/// Whether `address` is canonical for a linear-address width of `bits`: every
/// bit from the width's top bit upward is a copy of it.
fn canonical(address: u64, bits: u8) -> bool {
    let unused = u64::BITS.saturating_sub(u32::from(bits)).min(u64::BITS - 1);
    ((address << unused).cast_signed() >> unused).cast_unsigned() == address
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::{GuestSegment, GuestState, GuestTable, Limits, Rule, Subject, Violation, check};
    use crate::VmEntry;

    /// A 64-bit code segment: execute/read/accessed, ring 0, long mode.
    const CODE: u32 = 0xA09B;
    /// A flat writable data segment: read/write/accessed, ring 0, page
    /// granular.
    const DATA: u32 = 0xC093;
    /// A busy 64-bit task-state segment.
    const TASK: u32 = 0x8B;
    /// A segment marked unusable.
    const UNUSABLE: u32 = 1 << 16;

    /// A segment with `selector`, a flat limit and `rights`.
    fn flat(selector: u16, rights: u32) -> GuestSegment {
        GuestSegment {
            selector,
            base: 0,
            limit: 0xFFFF_FFFF,
            rights,
        }
    }

    /// The state of an ordinary 64-bit guest, which passes every check.
    fn valid() -> GuestState {
        GuestState {
            entry: VmEntry::IA32E_MODE_GUEST
                | VmEntry::LOAD_IA32_EFER
                | VmEntry::LOAD_DEBUG_CONTROLS,
            cr0: 0x8000_0031,
            cr3: 0x5F40_1000,
            cr4: 0x2668,
            dr7: 0x400,
            efer: 0xD00,
            rflags: 0x2,
            rip: 0x5361_56B0,
            sysenter_esp: 0,
            sysenter_eip: 0,
            es: flat(0x30, DATA),
            cs: flat(0x38, CODE),
            ss: flat(0x30, DATA),
            ds: flat(0x30, DATA),
            fs: flat(0x30, DATA),
            gs: flat(0x30, DATA),
            ldtr: flat(0, UNUSABLE),
            tr: GuestSegment {
                selector: 0x40,
                base: 0x5000,
                limit: 0x67,
                rights: TASK,
            },
            gdtr: GuestTable {
                base: 0x5F5F_0000,
                limit: 0x47,
            },
            idtr: GuestTable {
                base: 0x5F5F_1000,
                limit: 0xFFF,
            },
        }
    }

    /// A processor that forces `CR0.PE`, `CR0.PG`, `CR0.NE` and `CR4.VMXE`,
    /// with a 39-bit physical and 48-bit linear address width.
    fn limits() -> Limits {
        Limits {
            cr0_fixed0: 0x8000_0021,
            cr0_fixed1: 0xFFFF_FFFF,
            cr4_fixed0: 0x2000,
            cr4_fixed1: 0x3F_FFFF,
            physical_address_bits: 39,
            linear_address_bits: 48,
        }
    }

    /// Every violation `state` reports.
    fn violations(state: &GuestState) -> Vec<Violation> {
        let mut found = Vec::new();
        check(state, &limits(), |violation| found.push(violation));
        found
    }

    /// The subject and rule of each violation, which is what a test asserts.
    fn rules(state: &GuestState) -> Vec<(Subject, Rule)> {
        violations(state)
            .into_iter()
            .map(|violation| (violation.subject, violation.rule))
            .collect()
    }

    #[test]
    fn an_ordinary_64_bit_guest_passes() {
        assert_eq!(violations(&valid()), Vec::new());
    }

    #[test]
    fn a_null_task_register_is_unusable_and_refused() {
        let state = GuestState {
            tr: flat(0, UNUSABLE),
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Tr, Rule::Unusable)]);
    }

    #[test]
    fn an_available_task_state_segment_is_the_wrong_type() {
        let state = GuestState {
            tr: GuestSegment {
                rights: 0x89,
                ..valid().tr
            },
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Tr, Rule::Type)]);
    }

    #[test]
    fn a_data_segment_never_marked_accessed_is_the_wrong_type() {
        let state = GuestState {
            ds: flat(0x30, DATA & !1),
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Ds, Rule::Type)]);
    }

    #[test]
    fn unusable_data_segments_are_not_checked() {
        let state = GuestState {
            ds: flat(0, UNUSABLE),
            es: flat(0, UNUSABLE),
            fs: flat(0, UNUSABLE),
            gs: flat(0, UNUSABLE),
            ..valid()
        };
        assert_eq!(violations(&state), Vec::new());
    }

    #[test]
    fn a_long_code_segment_may_not_claim_the_32_bit_size() {
        let state = GuestState {
            cs: flat(0x38, CODE | 1 << 14),
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Cs, Rule::LongWithDefaultSize)]);
    }

    #[test]
    fn a_byte_granular_segment_may_not_reach_past_a_megabyte() {
        let state = GuestState {
            ds: flat(0x30, DATA & !(1 << 15)),
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Ds, Rule::Granularity)]);
    }

    #[test]
    fn the_stack_and_code_selectors_must_request_the_same_privilege() {
        let state = GuestState {
            ss: flat(0x33, DATA | 3 << 5),
            ..valid()
        };
        assert!(rules(&state).contains(&(Subject::Ss, Rule::RequestedPrivilege)));
    }

    #[test]
    fn efer_must_agree_with_the_64_bit_guest_control() {
        let state = GuestState {
            efer: 0x100,
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Efer, Rule::LongModeMismatch)]);
    }

    #[test]
    fn control_registers_must_honour_the_fixed_bits() {
        let state = GuestState {
            cr4: valid().cr4 & !0x2000,
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Cr4, Rule::FixedBits)]);
    }

    #[test]
    fn cr3_may_not_name_memory_beyond_the_physical_width() {
        let state = GuestState {
            cr3: 1 << 40,
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Cr3, Rule::BeyondPhysicalWidth)]);
    }

    #[test]
    fn a_64_bit_instruction_pointer_must_be_canonical() {
        let state = GuestState {
            rip: 0x0000_8000_0000_0000,
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Rip, Rule::NotCanonical)]);
    }

    #[test]
    fn rflags_must_keep_its_fixed_one_bit() {
        let state = GuestState {
            rflags: 0,
            ..valid()
        };
        assert_eq!(rules(&state), [(Subject::Rflags, Rule::Reserved)]);
    }

    #[test]
    fn a_violation_reads_as_subject_value_and_rule() {
        let violation = Violation {
            subject: Subject::Tr,
            rule: Rule::Unusable,
            value: 0x1_0000,
        };
        assert_eq!(
            std::format!("{violation}"),
            "tr 0x10000: unusable, which this register may not be"
        );
    }
}
