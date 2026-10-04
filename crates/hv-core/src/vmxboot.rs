//! A first attempt at running the captured firmware as an Intel VMX guest.
//!
//! The SVM path enters firmware as its first guest by copying the captured
//! firmware state into a VMCB and running it behind nested paging. This does
//! the Intel equivalent as far as it goes today: it enters VMX operation,
//! builds an extended page table that identity-maps all of physical memory,
//! programs a VMCS from the same captured firmware state — reconciling the
//! control registers against the bits VMX forces and converting each segment's
//! attributes to the access-rights word a VMCS wants — and enters the guest,
//! reporting exactly how the entry and the first exits went.
//!
//! It is a bring-up probe, not the finished backend. Firmware's captured
//! instruction pointer is deliberately zero — the save area leaves it for
//! whoever enters the guest to fill — so there is no portal to resume into yet;
//! instead the guest begins at a one-instruction stub of citrine's own that
//! calls straight back into the host. What that establishes is the hard,
//! machine-specific core: whether firmware's real control registers, segments
//! and descriptor tables pass the VM-entry consistency checks on this
//! processor, and whether the world switch and the EPT carry firmware's own
//! paging. Before the entry it checks the programmed state against the VM-entry
//! rules itself, logging each one broken, because a processor that refuses a
//! guest's state does not say which rule it failed; after a refusal it logs the
//! state the processor rejected. That is what the next step — a full portal and
//! guest memory — is built from.
//!
//! Two parts of firmware's state are adjusted on the way in, because VMX
//! demands what AMD's world switch does not check: a null task register becomes
//! a minimal busy 64-bit one, and every loaded code or data segment is marked
//! accessed.
//!
//! Two more keep the stub, which is citrine's code and not firmware's, from
//! being diverted into firmware's handlers. The stub is entered with interrupts
//! masked, because an interrupt firmware's timer left pending would otherwise
//! be delivered through firmware's descriptor table before the stub's first
//! instruction. And every exception exits to the host instead of being
//! delivered, so a fault taken running the stub — a page firmware maps
//! non-executable, say — is reported with its vector, error code and address
//! rather than vanishing into firmware's exception handler.

use alloc::{boxed::Box, vec::Vec};

use ept::{ENTRIES, Memory};
use log::{error, info};
use paging::AddressSpace;
use snapshot::FirmwareContext;
use svm::{SaveArea, Segment, SegmentAttributes};
use vmcs::{Registers, VmFail, Vmcs, controls, fixed, host, inspect, instr};
use vmexits::{Exit, Partition, Stop};
use vmx::{
    AccessRights, BasicExitReason, EptEntry, Field, FieldEncoding, Interruption, PAGE_BYTES,
    check::{self, GuestState, Subject},
    segment::{BUSY_TSS_TYPE, TYPE_ACCESSED},
};
use x86_64::{VirtAddr, registers::rflags::RFlags};

/// Bytes in a gibibyte, the unit the EPT identity map is sized in.
const GIB: u64 = 1 << 30;

/// The limit of the smallest 64-bit task-state segment, which is what a guest
/// given a task register of citrine's own is described with.
const TSS_LIMIT: u32 = 0x67;

/// An exception bitmap with every vector set, so every exception the guest
/// takes exits instead of being delivered.
const EVERY_EXCEPTION: u64 = 0xFFFF_FFFF;

/// The four VMCS fields that describe one guest segment: selector, base, limit
/// and access rights.
type SegmentFields = (FieldEncoding, FieldEncoding, FieldEncoding, FieldEncoding);

/// A page-aligned, page-sized region: a VMXON region, a VMCS, or an EPT frame.
#[repr(C, align(4096))]
struct Page([u8; PAGE_BYTES]);

impl Page {
    /// A fresh zeroed page on the heap, inside the chunk the direct map covers
    /// so its physical address is reachable.
    fn zeroed() -> Box<Self> {
        Box::new(Self([0; PAGE_BYTES]))
    }
}

/// EPT table frames, allocated from the heap and translated to the physical
/// addresses the processor walks.
struct EptFrames<'a> {
    /// Each allocated frame, kept alive here, paired with its physical address.
    frames: Vec<(Box<Page>, u64)>,
    /// The address space the frames are translated through.
    space: &'a AddressSpace,
}

impl Memory for EptFrames<'_> {
    fn allocate(&mut self) -> Option<u64> {
        let frame = Page::zeroed();
        let virt = VirtAddr::new(core::ptr::from_ref(&frame.0).addr() as u64);
        let phys = self.space.translate(virt).ok()?.as_u64();
        self.frames.push((frame, phys));
        Some(phys)
    }

    fn table(&mut self, phys: u64) -> &mut [EptEntry; ENTRIES] {
        let frame = self
            .frames
            .iter_mut()
            .find(|(_, at)| *at == phys)
            .expect("the mapper only names frames this Memory allocated");
        let page: *mut Page = core::ptr::from_mut(&mut *frame.0);
        // SAFETY: a `Page` is 4-KiB aligned and sized, exactly a table of
        // `ENTRIES` eight-byte entries; it is owned here and reached uniquely.
        unsafe { &mut *page.cast::<[EptEntry; ENTRIES]>() }
    }
}

/// A partition that resolves no fault: the identity EPT maps all of physical
/// memory, so a violation is a real fault worth stopping on rather than mapping
/// over.
struct Identity;

impl Partition for Identity {
    fn fault(&mut self, _gpa: u64) -> bool {
        false
    }
}

/// A one-instruction guest: `VMCALL` straight back into the host, then halt.
///
/// Firmware's captured instruction pointer is zero, so the guest needs an entry
/// point of citrine's own. This is the smallest one that produces a clean,
/// recognizable exit: reaching the host through its `VMCALL` is proof the
/// firmware state was entered and the first instruction fetched and run.
#[unsafe(naked)]
unsafe extern "C" fn firmware_stub() {
    core::arch::naked_asm!("vmcall", "2:", "hlt", "jmp 2b");
}

/// Enters the captured firmware as a VMX guest and reports the outcome.
///
/// Always returns, having logged how far it got; the caller halts afterward,
/// because this is a probe and there is no guest loop to stay in yet.
pub(crate) fn attempt(space: &AddressSpace, firmware: &FirmwareContext, top_of_ram: u64) {
    info!("vmxboot: entering VMX to run the captured firmware as a guest");

    let mut vmxon = Page::zeroed();
    let mut vmcs = Page::zeroed();
    let (Ok(vmxon_phys), Ok(vmcs_phys)) = (
        space.translate(VirtAddr::new(core::ptr::from_ref(&vmxon.0).addr() as u64)),
        space.translate(VirtAddr::new(core::ptr::from_ref(&vmcs.0).addr() as u64)),
    ) else {
        error!("vmxboot: could not translate the VMX regions to physical addresses");
        return;
    };

    // SAFETY: the two pages are freshly allocated, exclusively owned, their
    // physical addresses just translated from their own virtual ones, and this
    // runs in long mode with paging on; both live to the end of this function.
    let vmx = match unsafe { vmcs::enter(vmxon.0.as_mut_ptr().cast(), vmxon_phys) } {
        Ok(vmx) => vmx,
        Err(error) => {
            error!("vmxboot: could not enter VMX operation: {error}");
            return;
        }
    };

    // SAFETY: this processor is in VMX operation, the VMCS page is exclusively
    // owned, and `vmx.basic()` is this processor's own.
    let mut cell =
        match unsafe { Vmcs::activate(vmcs.0.as_mut_ptr().cast(), vmcs_phys, vmx.basic()) } {
            Ok(cell) => cell,
            Err(error) => {
                error!("vmxboot: could not make a VMCS current: {error}");
                // SAFETY: in VMX operation with no current VMCS.
                unsafe { leave() };
                return;
            }
        };

    let gibibytes = usize::try_from(top_of_ram.div_ceil(GIB)).unwrap_or(usize::MAX);
    let mut memory = EptFrames {
        frames: Vec::new(),
        space,
    };
    let eptp = match ept::identity(&mut memory, gibibytes) {
        Ok(pointer) => pointer,
        Err(error) => {
            error!("vmxboot: could not build the EPT over {gibibytes} GiB: {error:?}");
            cleanup(&cell);
            return;
        }
    };
    info!("vmxboot: EPT identity-maps {gibibytes} GiB of physical memory");

    let Ok(entry) = space.translate(VirtAddr::new((firmware_stub as *const ()).addr() as u64))
    else {
        error!("vmxboot: could not translate the guest entry stub");
        cleanup(&cell);
        return;
    };

    // SAFETY: `cell` is the current VMCS; host and control programming read this
    // processor's own state and the capability registers it has in VMX
    // operation, and the EPT built above identity-maps the firmware's memory.
    // The guest begins at the stub's physical address, which firmware's own
    // identity-mapping page tables and the identity EPT both resolve to itself.
    let programmed = unsafe {
        host::program(&cell)
            .and_then(|()| controls::program(&cell, Some(eptp)))
            .and_then(|()| cell.write(Field::EXCEPTION_BITMAP, EVERY_EXCEPTION))
            .and_then(|()| program_firmware(&cell, &firmware.cpu, entry.as_u64()))
    };
    if let Err(error) = programmed {
        error!("vmxboot: could not program the VMCS: {error}");
        cleanup(&cell);
        return;
    }
    info!(
        "vmxboot: VMCS programmed from firmware state; guest cr3 {:#x}, entry {:#x}, firmware rflags {:#x} (entered with interrupts masked)",
        firmware.cpu.cr3,
        entry.as_u64(),
        firmware.cpu.rflags
    );

    // SAFETY: `cell` is the current, fully programmed VMCS, in VMX operation.
    unsafe { predict(&cell) };

    let mut registers = Registers::default();
    // SAFETY: `cell` is the current, fully programmed VMCS, this processor is in
    // VMX operation, and bring-up installed the general-protection vector.
    let outcome = unsafe { vmexits::run(&mut cell, &mut registers, &mut Identity) };
    report(&cell, outcome);

    cleanup(&cell);
    // SAFETY: `cell` is no longer current after `cleanup`, the precondition for
    // leaving VMX operation.
    unsafe { leave() };
    // The EPT frames are walked by the processor for the whole run above, so they
    // are dropped only now.
    drop(memory);
    info!("vmxboot: probe complete");
}

/// Programs the guest half of the current VMCS from a captured firmware save
/// area, beginning at `entry`.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation, with
/// the host state and controls programmed around it.
unsafe fn program_firmware(cell: &Vmcs, save: &SaveArea, entry: u64) -> Result<(), VmFail> {
    let cr0 = reconcile_cr(
        save.cr0,
        fixed::IA32_VMX_CR0_FIXED0,
        fixed::IA32_VMX_CR0_FIXED1,
    );
    let cr4 = reconcile_cr(
        save.cr4,
        fixed::IA32_VMX_CR4_FIXED0,
        fixed::IA32_VMX_CR4_FIXED1,
    );

    // SAFETY: the caller guarantees the current VMCS; the read shadows carry the
    // same reconciled control registers, which a valid VMCS requires.
    unsafe {
        cell.write(Field::GUEST_CR0, cr0)?;
        cell.write(Field::GUEST_CR3, save.cr3)?;
        cell.write(Field::GUEST_CR4, cr4)?;
        cell.write(Field::CR0_READ_SHADOW, cr0)?;
        cell.write(Field::CR4_READ_SHADOW, cr4)?;
        cell.write(Field::GUEST_IA32_EFER, save.efer)?;
        cell.write(Field::GUEST_DR7, save.dr7)?;
        cell.write(Field::GUEST_RFLAGS, quiet(save.rflags))?;
        cell.write(Field::GUEST_RIP, entry)?;
        cell.write(Field::GUEST_RSP, save.rsp)?;
        program_segments(cell, save)?;
        cell.write(Field::GUEST_GDTR_BASE, save.gdtr.base)?;
        cell.write(Field::GUEST_GDTR_LIMIT, u64::from(save.gdtr.limit))?;
        cell.write(Field::GUEST_IDTR_BASE, save.idtr.base)?;
        cell.write(Field::GUEST_IDTR_LIMIT, u64::from(save.idtr.limit))?;
        cell.write(Field::GUEST_IA32_SYSENTER_CS, save.sysenter_cs)?;
        cell.write(Field::GUEST_IA32_SYSENTER_ESP, save.sysenter_esp)?;
        cell.write(Field::GUEST_IA32_SYSENTER_EIP, save.sysenter_eip)?;
        cell.write(Field::GUEST_INTERRUPTIBILITY_STATE, 0)?;
        cell.write(Field::GUEST_ACTIVITY_STATE, 0)?;
        cell.write(Field::GUEST_PENDING_DBG_EXCEPTIONS, 0)?;
    }
    Ok(())
}

/// Firmware's `RFLAGS` with interrupts masked, for entering the stub.
///
/// Firmware runs with interrupts enabled, and its timer keeps raising them
/// while the host runs with them off, so one is pending at entry. Delivered,
/// it would run firmware's handler in place of the stub's first instruction.
fn quiet(rflags: u64) -> u64 {
    rflags & !RFlags::INTERRUPT_FLAG.bits()
}

/// Writes all eight guest segments from a captured save area.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor.
unsafe fn program_segments(cell: &Vmcs, save: &SaveArea) -> Result<(), VmFail> {
    let segments: [(SegmentFields, &Segment); 8] = [
        (
            fields_of(
                Field::GUEST_ES_SELECTOR,
                Field::GUEST_ES_BASE,
                Field::GUEST_ES_LIMIT,
                Field::GUEST_ES_ACCESS_RIGHTS,
            ),
            &save.es,
        ),
        (
            fields_of(
                Field::GUEST_CS_SELECTOR,
                Field::GUEST_CS_BASE,
                Field::GUEST_CS_LIMIT,
                Field::GUEST_CS_ACCESS_RIGHTS,
            ),
            &save.cs,
        ),
        (
            fields_of(
                Field::GUEST_SS_SELECTOR,
                Field::GUEST_SS_BASE,
                Field::GUEST_SS_LIMIT,
                Field::GUEST_SS_ACCESS_RIGHTS,
            ),
            &save.ss,
        ),
        (
            fields_of(
                Field::GUEST_DS_SELECTOR,
                Field::GUEST_DS_BASE,
                Field::GUEST_DS_LIMIT,
                Field::GUEST_DS_ACCESS_RIGHTS,
            ),
            &save.ds,
        ),
        (
            fields_of(
                Field::GUEST_FS_SELECTOR,
                Field::GUEST_FS_BASE,
                Field::GUEST_FS_LIMIT,
                Field::GUEST_FS_ACCESS_RIGHTS,
            ),
            &save.fs,
        ),
        (
            fields_of(
                Field::GUEST_GS_SELECTOR,
                Field::GUEST_GS_BASE,
                Field::GUEST_GS_LIMIT,
                Field::GUEST_GS_ACCESS_RIGHTS,
            ),
            &save.gs,
        ),
        (
            fields_of(
                Field::GUEST_LDTR_SELECTOR,
                Field::GUEST_LDTR_BASE,
                Field::GUEST_LDTR_LIMIT,
                Field::GUEST_LDTR_ACCESS_RIGHTS,
            ),
            &save.ldtr,
        ),
        (
            fields_of(
                Field::GUEST_TR_SELECTOR,
                Field::GUEST_TR_BASE,
                Field::GUEST_TR_LIMIT,
                Field::GUEST_TR_ACCESS_RIGHTS,
            ),
            &task_register(&save.tr),
        ),
    ];
    for (fields, segment) in segments {
        // SAFETY: the caller guarantees the current VMCS.
        unsafe { write_segment(cell, fields, segment)? };
    }
    Ok(())
}

/// The task register the guest enters with.
///
/// A 64-bit VM entry insists on a usable, busy 64-bit task-state segment, but
/// firmware need never load one: code running at ring 0 with no
/// interrupt-stack-table entries never consults it, and the AMD world switch
/// does not check, so the captured task register can be null. When it is, the
/// guest gets the task register a processor holds out of reset — selector and
/// base zero — typed as the busy 64-bit segment long mode requires and given
/// the smallest limit such a segment has. Firmware never reads through it, and
/// an operating system loads its own before anything could. A task register
/// firmware did load is kept as it was.
fn task_register(captured: &Segment) -> Segment {
    if captured.attributes.present() {
        return *captured;
    }
    Segment {
        selector: captured.selector,
        attributes: SegmentAttributes::new()
            .with_kind(BUSY_TSS_TYPE)
            .with_present(true),
        limit: TSS_LIMIT,
        base: 0,
    }
}

/// Groups a segment's four field encodings, so a caller names each segment
/// once.
const fn fields_of(
    selector: FieldEncoding,
    base: FieldEncoding,
    limit: FieldEncoding,
    rights: FieldEncoding,
) -> SegmentFields {
    (selector, base, limit, rights)
}

/// Writes one guest segment's four fields from a captured [`Segment`].
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor.
unsafe fn write_segment(
    cell: &Vmcs,
    fields: SegmentFields,
    segment: &Segment,
) -> Result<(), VmFail> {
    let (selector, base, limit, rights) = fields;
    // SAFETY: the caller guarantees the current VMCS.
    unsafe {
        cell.write(selector, u64::from(segment.selector))?;
        cell.write(base, segment.base)?;
        cell.write(limit, u64::from(segment.limit))?;
        cell.write(rights, access_rights(segment))?;
    }
    Ok(())
}

/// The VMX access-rights word for a captured segment.
///
/// The save area and the VMCS carry the same fields — type, descriptor and
/// present bits, privilege level, the available/long/default-size/granularity
/// bits — but lay them out differently, and the VMCS has an extra *unusable*
/// bit the save area signals instead by clearing the present bit. Rebuilding
/// the word field by field, rather than shifting bits, keeps the two layouts
/// from ever being confused.
///
/// A loaded code or data segment is always marked accessed: loading it sets
/// the bit, and VM entry refuses one without it. The captured attributes come
/// from the descriptor in firmware's table rather than from the processor, so
/// a table rewritten after the load can have lost the bit the segment
/// register still carries.
fn access_rights(segment: &Segment) -> u64 {
    let attributes = segment.attributes;
    if !attributes.present() {
        return u64::from(AccessRights::new().with_unusable(true).into_bits());
    }
    let kind = if attributes.descriptor() {
        attributes.kind() | TYPE_ACCESSED
    } else {
        attributes.kind()
    };
    let rights = AccessRights::new()
        .with_kind(kind)
        .with_descriptor(attributes.descriptor())
        .with_dpl(attributes.dpl())
        .with_present(true)
        .with_available(attributes.available())
        .with_long(attributes.long())
        .with_default_size(attributes.default_size())
        .with_granularity(attributes.granularity());
    u64::from(rights.into_bits())
}

/// A control register reconciled against its fixed-bit pair, so VM entry
/// accepts it. The two fixed-bit registers always exist in VMX operation; the
/// fallbacks force no bit on and forbid none, leaving the value untouched.
fn reconcile_cr(value: u64, fixed0: u32, fixed1: u32) -> u64 {
    let low = probe::read(fixed0).unwrap_or(0);
    let high = probe::read(fixed1).unwrap_or(!0);
    fixed::reconcile(value, low, high)
}

/// Logs how the guest entry and its first exits went.
fn report(cell: &Vmcs, outcome: Exit) {
    if let Exit::Stopped(_) = outcome {
        // SAFETY: `cell` is current, in VMX operation.
        unsafe { locate(cell) };
    }
    match outcome {
        Exit::Vmcall => info!(
            "vmxboot: the firmware guest reached its VMCALL; VM entry and the world switch carried the firmware state"
        ),
        Exit::Stopped(Stop::EntryRejected(fail)) => {
            // SAFETY: `cell` is current; the instruction-error field records why
            // the entry was rejected.
            let number = unsafe { cell.read(Field::VM_INSTRUCTION_ERROR) }.unwrap_or(0);
            error!("vmxboot: VM entry was rejected ({fail}); VM-instruction-error {number}");
        }
        Exit::Stopped(Stop::EntryFailure(reason)) => {
            // SAFETY: `cell` is current; after a failed entry the exit
            // qualification says whether a specific cause was recorded.
            let qualification = unsafe { cell.read(Field::EXIT_QUALIFICATION) }.unwrap_or(0);
            error!(
                "vmxboot: VM entry failed on the guest state ({reason}, qualification {qualification})"
            );
            if reason == BasicExitReason::ENTRY_FAILURE_GUEST_STATE {
                // SAFETY: `cell` is current, in VMX operation.
                unsafe { dump(cell) };
            }
        }
        Exit::Stopped(Stop::EptViolation(violation)) => {
            // SAFETY: `cell` is current; the faulting guest-physical address is
            // readable after an EPT violation.
            let gpa = unsafe { cell.read(Field::GUEST_PHYSICAL_ADDRESS) }.unwrap_or(0);
            error!("vmxboot: the firmware guest took an EPT violation at {gpa:#x}: {violation:?}");
        }
        Exit::Stopped(Stop::Unhandled(BasicExitReason::EXCEPTION_OR_NMI)) => {
            // SAFETY: `cell` is current, in VMX operation.
            unsafe { exception(cell) };
        }
        Exit::Stopped(stop) => error!("vmxboot: the firmware guest stopped: {stop:?}"),
    }
}

/// Logs where the guest was when it stopped.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
unsafe fn locate(cell: &Vmcs) {
    // SAFETY: the caller guarantees the current VMCS; these guest-state fields
    // are readable after any exit.
    let (rip, rsp, rflags) = unsafe {
        (
            cell.read(Field::GUEST_RIP).unwrap_or(0),
            cell.read(Field::GUEST_RSP).unwrap_or(0),
            cell.read(Field::GUEST_RFLAGS).unwrap_or(0),
        )
    };
    error!("vmxboot: the guest stopped at rip {rip:#x}, rsp {rsp:#x}, rflags {rflags:#x}");
}

/// Logs the exception an exception exit intercepted: its vector and kind, its
/// error code, and the exit qualification, which for a page fault is the
/// linear address that faulted.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation, and its
/// last exit an exception or non-maskable interrupt.
unsafe fn exception(cell: &Vmcs) {
    // SAFETY: the caller guarantees the current VMCS and an exception exit,
    // after which these three fields describe the event.
    let (info, code, qualification) = unsafe {
        (
            cell.read(Field::VM_EXIT_INTERRUPTION_INFO).unwrap_or(0),
            cell.read(Field::VM_EXIT_INTERRUPTION_ERROR_CODE)
                .unwrap_or(0),
            cell.read(Field::EXIT_QUALIFICATION).unwrap_or(0),
        )
    };
    let event = Interruption::from_bits(u32::try_from(info).unwrap_or(0));
    let code = if event.has_error_code() { code } else { 0 };
    error!(
        "vmxboot: the guest took exception vector {} ({:?}), error code {code:#x}, qualification {qualification:#x}",
        event.vector(),
        event.kind()
    );
}

/// Checks the programmed guest state against the VM-entry rules before the
/// entry is tried, logging each one it breaks.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
unsafe fn predict(cell: &Vmcs) {
    // SAFETY: the caller guarantees the current VMCS and VMX operation.
    let (state, limits) = match unsafe { inspect::guest_state(cell) } {
        // SAFETY: as above; VMX operation means the fixed-bit registers exist.
        Ok(state) => (state, unsafe { inspect::limits() }),
        Err(error) => {
            error!("vmxboot: could not read the guest state back: {error}");
            return;
        }
    };
    let mut broken = 0_usize;
    check::check(&state, &limits, |violation| {
        broken += 1;
        error!("vmxboot: guest state breaks a VM-entry rule: {violation}");
    });
    if broken == 0 {
        info!("vmxboot: the guest state passes every VM-entry check citrine restates");
    }
}

/// Logs the guest state a refused entry left in the VMCS, raw, so a check this
/// does not restate can still be found from the numbers.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
unsafe fn dump(cell: &Vmcs) {
    // SAFETY: the caller guarantees the current VMCS and VMX operation.
    let state: GuestState = match unsafe { inspect::guest_state(cell) } {
        Ok(state) => state,
        Err(error) => {
            error!("vmxboot: could not read the guest state back: {error}");
            return;
        }
    };
    error!(
        "vmxboot: guest cr0 {:#x} cr3 {:#x} cr4 {:#x} efer {:#x} rflags {:#x} rip {:#x}",
        state.cr0, state.cr3, state.cr4, state.efer, state.rflags, state.rip
    );
    error!(
        "vmxboot: guest gdtr {:#x}+{:#x} idtr {:#x}+{:#x} dr7 {:#x} entry controls {:#x}",
        state.gdtr.base,
        state.gdtr.limit,
        state.idtr.base,
        state.idtr.limit,
        state.dr7,
        state.entry.bits()
    );
    for (name, segment) in [
        (Subject::Cs, state.cs),
        (Subject::Ss, state.ss),
        (Subject::Ds, state.ds),
        (Subject::Es, state.es),
        (Subject::Fs, state.fs),
        (Subject::Gs, state.gs),
        (Subject::Ldtr, state.ldtr),
        (Subject::Tr, state.tr),
    ] {
        error!(
            "vmxboot: guest {name} selector {:#06x} base {:#x} limit {:#x} rights {:#x}",
            segment.selector, segment.base, segment.limit, segment.rights
        );
    }
}

/// Clears the current VMCS, leaving none current.
fn cleanup(cell: &Vmcs) {
    // SAFETY: `cell` is the current VMCS, so clearing it leaves none current.
    unsafe {
        let _ = instr::vmclear(cell.region()).ok();
    }
}

/// Leaves VMX operation, logging if the instruction is refused.
///
/// # Safety
///
/// This processor must be in VMX operation with no current VMCS.
unsafe fn leave() {
    // SAFETY: the caller guarantees VMX operation with no current VMCS.
    if let Err(error) = unsafe { instr::vmxoff() }.ok() {
        error!("vmxboot: VMXOFF failed: {error}");
    }
}
