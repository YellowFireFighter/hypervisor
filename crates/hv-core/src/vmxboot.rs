//! Running the captured firmware as an Intel VMX guest, entered at the portal.
//!
//! The SVM path enters firmware as its first guest by copying the captured
//! firmware state into a VMCB, running it behind nested paging, and resuming it
//! not where firmware stopped but at the portal — a page of citrine's own that
//! starts the operating system's boot manager and replaces `ExitBootServices`
//! with a wrapper, so the host learns the moment firmware's services end. This
//! is the Intel equivalent as far as it goes today: it enters VMX operation,
//! builds an extended page table that identity-maps all of physical memory,
//! programs a VMCS from the captured firmware state — reconciling the control
//! registers against the bits VMX forces and converting each segment's packed
//! attributes to the access-rights word a VMCS wants — resumes firmware at the
//! portal, and answers the portal's notifications as the guest leaves firmware.
//!
//! It is still a bring-up probe. The portal starts whatever boot manager the
//! loader preloaded; built with none (the `no-guest` loader feature) its
//! `StartImage` returns and the probe reports that rather than an OS running.
//! The post-`ExitBootServices` VMX path — starting the other processors,
//! concealing the portal, interposing on devices — is not built yet, so a guest
//! that gets that far is reported and stopped. What this establishes is the
//! hard, machine-specific core: that firmware's real state passes the VM-entry
//! checks, that the world switch and the EPT carry firmware's own paging, and
//! that firmware resumes into and runs citrine's own portal on an Intel
//! machine. Before entry it checks the programmed state against the VM-entry
//! rules and logs any it breaks; after a refusal it logs the rejected state.
//!
//! Two parts of firmware's state are adjusted on the way in, because VMX
//! demands what AMD's world switch does not check: a null task register becomes
//! a minimal busy 64-bit one, and every loaded code or data segment is marked
//! accessed. Firmware keeps its interrupts enabled, so its own handlers run as
//! the guest, and the guest's `CR3` accesses are kept from exiting where the
//! processor's "true" capability registers permit it, since firmware saves
//! `CR3` on every interrupt entry.

use alloc::{boxed::Box, vec::Vec};

use apic::LocalState;
use ept::{ENTRIES, Memory};
use handoff::Handoff;
use log::{error, info, warn};
use paging::{AddressSpace, CacheType, Mapping, Protection};
use portal::{Notification, Portal};
use snapshot::FirmwareContext;
use svm::{SaveArea, Segment, SegmentAttributes};
use vmcs::{Registers, VmFail, Vmcs, controls, fixed, host, inspect, instr};
use vmexits::{ApicWrites, Exit, Partition, Stop};
use vmx::{
    AccessRights, BasicExitReason, EptAccess, EptEntry, EptMemoryType, Field, FieldEncoding,
    Interruption, PAGE_BYTES,
    check::{self, GuestState, Subject},
    segment::{BUSY_TSS_TYPE, TYPE_ACCESSED},
};
use x86_64::{PhysAddr, VirtAddr};

/// Bytes in a gibibyte, the unit the EPT identity map is sized in.
const GIB: u64 = 1 << 30;

/// The limit of the smallest 64-bit task-state segment, which is what a guest
/// given a task register of citrine's own is described with.
const TSS_LIMIT: u32 = 0x67;

/// Bit 10 of `IA32_APIC_BASE`: the APIC is in x2APIC mode, reached through
/// model-specific registers rather than the memory-mapped page.
const X2APIC_ENABLE: u64 = 1 << 10;

/// The alignment the portal's stack is realigned to. Firmware was captured
/// part-way through a call, so its stack pointer need not be aligned, and the
/// portal makes calls of its own that the ABI requires a 16-byte stack for.
const CALL_STACK_ALIGN: u64 = 16;

/// How long the preemption timer runs before forcing an exit, in the units the
/// processor counts it in. Long, so a guest making progress runs a substantial
/// stretch between forced exits rather than being sampled to a crawl: the
/// forced exit is only a backstop against a guest that would otherwise never
/// exit, and firmware running its boot manager needs real runtime to get
/// anywhere. A guest that is genuinely stuck still exits here and is caught.
const PREEMPTION_QUANTUM: u32 = 0x40_0000;

/// How many quanta the guest may sit at one instruction pointer before it is
/// declared stuck there.
const SPIN_THRESHOLD: u32 = 1000;

/// The most preemption quanta the probe lets the guest run before stopping
/// regardless. High, because reaching the boot manager's `ExitBootServices` is
/// the goal and firmware takes real time to get there; this is only a backstop
/// so a guest looping over many addresses forever still halts with a report
/// rather than requiring the machine to be power-cycled.
const SAMPLE_BUDGET: u32 = 100_000;

/// The most distinct instruction pointers the sampler logs, so a guest making
/// steady progress does not flood the log.
const SAMPLE_LOG_LIMIT: u32 = 24;

/// How often, in preemption quanta, to log a heartbeat once the distinct-rip
/// log is full, so a long run shows it is still progressing and where.
const HEARTBEAT_QUANTA: u32 = 1000;

/// Bit 16 of a local-vector-table entry: the interrupt is masked.
const LVT_MASKED: u32 = 1 << 16;

/// Bit 9 of `RFLAGS`: maskable interrupts are enabled.
const RFLAGS_INTERRUPT_ENABLE: u64 = 1 << 9;

/// How far the virtual timer's initial count is shifted to get its per-quantum
/// step, so it counts down over roughly this many quanta.
const TIMER_STEP_SHIFT: u32 = 4;

/// The lowest vector the local APIC delivers. A local-vector-table entry
/// programmed with a vector below this raises an illegal-vector error instead
/// of delivering an interrupt, so such an entry is not injected — doing so
/// would feed the guest an interrupt its real APIC never would.
const MIN_DELIVERABLE_VECTOR: u32 = 0x10;

/// How many bytes of the instruction stream to read back from a spinning
/// guest, enough to capture a tight firmware poll loop and decode what it
/// waits on.
const SPIN_CODE_BYTES: u64 = 32;

/// Byte offset of the end-of-interrupt register in the APIC page. The one
/// virtualized-APIC write whose effect the host completes on the real
/// controller, so interrupt delivery does not stall after the first interrupt.
const APIC_END_OF_INTERRUPT: u32 = 0xB0;

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

/// Completes the firmware guest's end-of-interrupt on the real local APIC.
///
/// The guest's APIC is virtualized, so its end-of-interrupt writes land in the
/// virtual-APIC page and never reach the real controller. Left there, the real
/// local APIC's in-service state never clears and it stops delivering the
/// periodic interrupt firmware's event loop runs on, which is what freezes a
/// firmware guest that is otherwise making progress. This forwards that one
/// write — and only it, so the inter-processor interrupts the virtualization
/// exists to swallow still stay in the page.
struct ApicForward {
    /// The real local APIC's end-of-interrupt register, mapped for the host in
    /// xAPIC mode; `None` leaves every write in the virtual page, as when the
    /// controller is in x2APIC mode and reached through model-specific
    /// registers this does not map.
    eoi: Option<*mut u32>,
}

impl ApicWrites for ApicForward {
    fn wrote(&mut self, offset: u32) {
        if offset != APIC_END_OF_INTERRUPT {
            return;
        }
        if let Some(eoi) = self.eoi {
            // SAFETY: `eoi` addresses the live uncached mapping of the real
            // local APIC's end-of-interrupt register; writing there completes
            // the interrupt the guest just finished, and the guest is not
            // running during this exit, so nothing races the write.
            unsafe { eoi.write_volatile(0) };
        }
    }
}

/// Maps the real local APIC so the host can complete the firmware guest's
/// end-of-interrupt on it, returning the mapping to keep alive for the run.
///
/// Only in xAPIC mode, where the controller is reached through this page.
/// x2APIC reaches it through model-specific registers, which this does not map;
/// there the guest's end-of-interrupt stays in the virtual page, as it did
/// before.
fn map_apic_eoi(space: &mut AddressSpace, firmware: &FirmwareContext) -> Option<Mapping> {
    if firmware.interrupts.base & X2APIC_ENABLE != 0 {
        return None;
    }
    let base = firmware.interrupts.base & !(PAGE_BYTES as u64 - 1);
    let Ok(phys) = PhysAddr::try_new(base) else {
        warn!("vmxboot: the APIC base {base:#x} is not a physical address to map");
        return None;
    };
    // SAFETY: `base` is the local APIC's register page, MMIO firmware owns and
    // this processor already reaches; mapping it read-write and uncached lets
    // the host complete end-of-interrupts on the real controller.
    let mapping = unsafe {
        space.map_physical(
            phys,
            PAGE_BYTES as u64,
            Protection::ReadWrite,
            CacheType::Uncached,
        )
    };
    match mapping {
        Ok(mapping) => Some(mapping),
        Err(error) => {
            warn!("vmxboot: could not map the local APIC to forward end-of-interrupt: {error:?}");
            None
        }
    }
}

/// Enters the captured firmware as a VMX guest and reports the outcome.
///
/// Always returns, having logged how far it got; the caller halts afterward,
/// because this is a probe and there is no guest loop to stay in yet.
pub(crate) fn attempt(space: &mut AddressSpace, firmware: &FirmwareContext, handoff: &Handoff) {
    info!("vmxboot: entering VMX to run the captured firmware as a guest");

    // Map the real local APIC up front, while the address space can still be
    // borrowed mutably, so the guest's end-of-interrupt can be completed on it.
    // Kept alive to the end of the run, because the forwarding reads through it.
    let lapic = map_apic_eoi(space, firmware);
    let apic_eoi = lapic
        .as_ref()
        .map(|mapping| (mapping.addr().as_u64() + u64::from(APIC_END_OF_INTERRUPT)) as *mut u32);
    // The mapping above is the only mutable use of the address space; everything
    // below reads it, so reborrow it shared for the rest of the run.
    let space: &AddressSpace = space;

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

    let gibibytes = usize::try_from(handoff.top_of_ram.div_ceil(GIB)).unwrap_or(usize::MAX);
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

    let portal = match Portal::place(space.direct_map(), handoff) {
        Ok(portal) => portal,
        Err(error) => {
            error!("vmxboot: could not place the portal: {error}");
            cleanup(&cell);
            return;
        }
    };
    let entry = portal.entry();

    // SAFETY: `cell` is the current VMCS; host and control programming read this
    // processor's own state and the capability registers it has in VMX
    // operation, and the EPT built above identity-maps the firmware's memory.
    // The guest begins at the stub's physical address, which firmware's own
    // identity-mapping page tables and the identity EPT both resolve to itself.
    let programmed = unsafe {
        host::program(&cell)
            .and_then(|()| controls::program(&cell, Some(eptp)))
            .and_then(|()| program_firmware(&cell, &firmware.cpu, entry.as_u64()))
            .and_then(|()| cell.write(Field::GUEST_RSP, firmware.cpu.rsp & !(CALL_STACK_ALIGN - 1)))
    };
    if let Err(error) = programmed {
        error!("vmxboot: could not program the VMCS: {error}");
        cleanup(&cell);
        return;
    }
    info!(
        "vmxboot: VMCS programmed from firmware state; guest cr3 {:#x}, entry {:#x}, firmware rflags {:#x} (interrupts left as firmware had them)",
        firmware.cpu.cr3,
        entry.as_u64(),
        firmware.cpu.rflags
    );

    // SAFETY: `cell` is the current VMCS, in VMX operation, and `program` wrote
    // the primary controls this rewrites.
    match unsafe { controls::relax_cr3_exiting(&cell) } {
        Ok(true) => {
            info!("vmxboot: guest CR3 accesses no longer exit; firmware may use its own paging");
        }
        Ok(false) => {
            info!("vmxboot: this processor forces CR3-exiting on; firmware's CR3 saves will exit");
        }
        Err(error) => error!("vmxboot: could not relax CR3-exiting: {error}"),
    }

    // Virtualize the guest's local APIC, so its register accesses — the
    // inter-processor interrupts that reset the machine above all — reach a
    // virtual controller rather than the real one. Kept alive until the run
    // ends, because the processor reads the virtual-APIC page throughout it.
    let apic_pages = setup_apic(space, firmware, &mut memory, eptp.root(), &cell);
    let vapic_addr = apic_pages
        .as_ref()
        .map(|(vapic, _)| core::ptr::from_ref(&vapic.0).addr() as u64);

    // SAFETY: `cell` is the current, fully programmed VMCS, in VMX operation.
    unsafe { predict(&cell) };

    resume(&mut cell, &portal, vapic_addr, space, apic_eoi);

    cleanup(&cell);
    // SAFETY: `cell` is no longer current after `cleanup`, the precondition for
    // leaving VMX operation.
    unsafe { leave() };
    // The EPT frames are walked by the processor for the whole run above, so they
    // are dropped only now.
    drop(memory);
    info!("vmxboot: probe complete");
}

/// Drives the firmware guest from the portal, answering each notification until
/// it leaves firmware or stops.
///
/// The run loop returns on every `VMCALL`, which is how the portal speaks to
/// the host; each is answered, the instruction stepped over, and the guest
/// resumed, until a notification ends the probe or the guest stops another way.
fn resume(
    cell: &mut Vmcs,
    portal: &Portal,
    vapic_addr: Option<u64>,
    space: &AddressSpace,
    apic_eoi: Option<*mut u32>,
) {
    let mut registers = Registers::default();
    let mut partition = Identity;
    let mut apic = ApicForward { eoi: apic_eoi };
    let mut dumped = false;
    // Arm the preemption timer, so a guest that spins without ever exiting is
    // still forced out each quantum and the resume loop can see where it is.
    // SAFETY: `cell` is the current VMCS in VMX operation.
    if let Err(error) = unsafe { controls::set_preemption_timer(cell, PREEMPTION_QUANTUM) } {
        error!("vmxboot: could not arm the preemption timer: {error}");
    }
    let mut last_rip = u64::MAX;
    let mut same = 0_u32;
    let mut total = 0_u32;
    let mut logged = 0_u32;
    loop {
        // SAFETY: `cell` is the current, fully programmed VMCS, this processor
        // is in VMX operation, and bring-up installed the general-protection
        // vector the forwarding relies on; the loop preserves that each entry.
        match unsafe { vmexits::run(cell, &mut registers, &mut partition, &mut apic) } {
            Exit::Vmcall => {
                if !notified(cell, &registers, portal) {
                    break;
                }
            }
            Exit::Stopped(Stop::Unhandled(BasicExitReason::PREEMPTION_TIMER_EXPIRED)) => {
                if let Some(addr) = vapic_addr {
                    if !dumped {
                        // SAFETY: `addr` names the live, held virtual-APIC page,
                        // and the guest is not running during this exit, so
                        // reading it races nothing.
                        unsafe { dump_vapic(addr) };
                        dumped = true;
                    }
                    // SAFETY: as above; `cell` is current, so the guest state
                    // the injection consults is readable and writable.
                    unsafe { drive_timer(cell, addr) };
                }
                if sample(
                    cell,
                    space,
                    &registers,
                    &mut last_rip,
                    &mut same,
                    &mut total,
                    &mut logged,
                ) {
                    break;
                }
            }
            outcome @ Exit::Stopped(_) => {
                report(cell, outcome);
                break;
            }
        }
    }
}

/// Ticks the guest's virtual-APIC timer and, when the guest can take it,
/// injects the timer interrupt.
///
/// The virtual-APIC page is memory, so its timer does not count on its own: a
/// guest waiting out a delay on the current-count register, or waiting for the
/// periodic timer's interrupt, would wait forever. Each quantum this decrements
/// the current count (reloading from the initial count when it runs out, as a
/// periodic timer does) and, if the timer is unmasked and the guest has
/// interrupts enabled and is not in an interrupt shadow, injects the vector the
/// timer entry names. It is a coarse timer — one tick per preemption quantum —
/// but it is one that moves, which is what a stuck guest needs.
///
/// # Safety
///
/// `addr` must name the live, page-sized virtual-APIC page with nothing else
/// writing it, and `cell` must be the current VMCS in VMX operation.
unsafe fn drive_timer(cell: &Vmcs, addr: u64) {
    // SAFETY: the caller guarantees a live page-sized region at `addr` that the
    // guest is not racing, and this writes only within it.
    let page = unsafe { core::slice::from_raw_parts_mut(addr as *mut u8, PAGE_BYTES) };
    let read = |page: &[u8], offset: usize| {
        u32::from_le_bytes([
            page[offset],
            page[offset + 1],
            page[offset + 2],
            page[offset + 3],
        ])
    };
    let lvt = read(page, APIC_LVT_TIMER);
    if lvt & LVT_MASKED != 0 {
        return;
    }
    let initial = read(page, APIC_TIMER_INITIAL_COUNT);
    let current = read(page, APIC_TIMER_CURRENT_COUNT);
    let step = (initial >> TIMER_STEP_SHIFT).max(1);
    let next = if current > step {
        current - step
    } else {
        initial
    };
    page[APIC_TIMER_CURRENT_COUNT..APIC_TIMER_CURRENT_COUNT + 4]
        .copy_from_slice(&next.to_le_bytes());

    // SAFETY: `cell` is current; these guest-state fields are readable, and the
    // entry interruption field is writable, after any exit.
    let (rflags, interruptibility) = unsafe {
        (
            cell.read(Field::GUEST_RFLAGS).unwrap_or(0),
            cell.read(Field::GUEST_INTERRUPTIBILITY_STATE).unwrap_or(0),
        )
    };
    if rflags & RFLAGS_INTERRUPT_ENABLE == 0 || interruptibility != 0 {
        return;
    }
    let vector = lvt & 0xFF;
    if vector < MIN_DELIVERABLE_VECTOR {
        return;
    }
    let vector = u8::try_from(vector).unwrap_or(0);
    let event = Interruption::inject(vector, vmx::event::Kind::External, false);
    // SAFETY: `cell` is current; the guest is interruptible, so an external
    // interrupt may be delivered on the next entry.
    let _ = unsafe { cell.write(Field::VM_ENTRY_INTERRUPTION_INFO, u64::from(event.bits())) };
}

/// Logs the virtual-APIC registers the guest left behind, to show what a stuck
/// guest is waiting on:/// Logs the virtual-APIC registers the guest left
/// behind, to show what a stuck guest is waiting on: a pending interrupt it
/// never takes, an in-service one it never finishes, or a timer it set and is
/// waiting out.
///
/// # Safety
///
/// `addr` must name the live, page-sized virtual-APIC page, and nothing else
/// may be writing it — which holds at a preemption exit, where the guest is not
/// running.
unsafe fn dump_vapic(addr: u64) {
    // SAFETY: the caller guarantees a live page-sized region at `addr`.
    let page = unsafe { core::slice::from_raw_parts(addr as *const u8, PAGE_BYTES) };
    let reg = |offset: usize| {
        u32::from_le_bytes([
            page[offset],
            page[offset + 1],
            page[offset + 2],
            page[offset + 3],
        ])
    };
    info!(
        "vmxboot: vAPIC tpr {:#x} ppr {:#x} spurious {:#x} lvt-timer {:#x} timer-init {:#x} timer-cur {:#x} divide {:#x}",
        reg(APIC_TASK_PRIORITY),
        reg(APIC_PROCESSOR_PRIORITY),
        reg(APIC_SPURIOUS),
        reg(APIC_LVT_TIMER),
        reg(APIC_TIMER_INITIAL_COUNT),
        reg(APIC_TIMER_CURRENT_COUNT),
        reg(APIC_TIMER_DIVIDE),
    );
    let bank =
        |base: usize| core::array::from_fn::<u32, 8, _>(|i| reg(base + i * APIC_REGISTER_STRIDE));
    info!("vmxboot: vAPIC isr {:#x?}", bank(APIC_IN_SERVICE));
    info!("vmxboot: vAPIC irr {:#x?}", bank(APIC_INTERRUPT_REQUEST));
    info!(
        "vmxboot: vAPIC icr {:#x}:{:#x} lint0 {:#x} lint1 {:#x} lvt-error {:#x}",
        reg(APIC_COMMAND_LOW),
        reg(APIC_COMMAND_HIGH),
        reg(APIC_LVT_LINT0),
        reg(APIC_LVT_LINT1),
        reg(APIC_LVT_ERROR),
    );
}

/// Records where a preemption-timer exit caught the guest and says whether the
/// probe should stop.
///
/// A guest that is making progress shows a changing instruction pointer, which
/// is logged as it moves; one that is stuck shows the same one over and over,
/// and once it has stood still for [`SPIN_THRESHOLD`] quanta it is reported as
/// spinning there and the probe stops. A guest doing neither after
/// [`SAMPLE_BUDGET`] quanta is stopped with what was seen, so the probe never
/// runs on forever.
fn sample(
    cell: &Vmcs,
    space: &AddressSpace,
    registers: &Registers,
    last_rip: &mut u64,
    same: &mut u32,
    total: &mut u32,
    logged: &mut u32,
) -> bool {
    // SAFETY: `cell` is current; the guest instruction pointer is readable after
    // any exit.
    let rip = unsafe { cell.read(Field::GUEST_RIP) }.unwrap_or(0);
    *total += 1;
    if rip == *last_rip {
        *same += 1;
        if *same >= SPIN_THRESHOLD {
            error!(
                "vmxboot: the guest is spinning at rip {rip:#x} ({same} quanta without moving); stopping"
            );
            spin_report(space, registers, rip);
            return true;
        }
    } else {
        *same = 0;
        *last_rip = rip;
        if *logged < SAMPLE_LOG_LIMIT {
            info!("vmxboot: guest running at rip {rip:#x} (quantum {total})");
            *logged += 1;
        }
    }
    if *logged >= SAMPLE_LOG_LIMIT && total.is_multiple_of(HEARTBEAT_QUANTA) {
        info!("vmxboot: still running at rip {rip:#x} after {total} quanta");
    }
    if *total >= SAMPLE_BUDGET {
        error!("vmxboot: sampled {total} quanta without a stop; last rip {rip:#x}");
        spin_report(space, registers, rip);
        return true;
    }
    false
}

/// Logs the register file and instruction bytes of a guest the sampler gave up
/// on, so the loop it is turning can be decoded from the host side.
///
/// A guest alternating between a couple of instruction pointers is in a tight
/// poll loop; the registers name what it is testing against and the bytes name
/// how. Firmware runs its own identity paging, where a linear address is its
/// physical address, so the instruction pointer doubles as the physical address
/// the direct map reads the code back from. A pointer the direct map cannot
/// reach, or one too high to be a physical address, is reported rather than
/// followed.
fn spin_report(space: &AddressSpace, registers: &Registers, rip: u64) {
    info!(
        "vmxboot: spin regs rax {:#x} rbx {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x} rbp {:#x}",
        registers.rax,
        registers.rbx,
        registers.rcx,
        registers.rdx,
        registers.rsi,
        registers.rdi,
        registers.rbp
    );
    info!(
        "vmxboot: spin regs r8 {:#x} r9 {:#x} r10 {:#x} r11 {:#x} r12 {:#x} r13 {:#x} r14 {:#x} r15 {:#x}",
        registers.r8,
        registers.r9,
        registers.r10,
        registers.r11,
        registers.r12,
        registers.r13,
        registers.r14,
        registers.r15
    );
    let Ok(phys) = PhysAddr::try_new(rip) else {
        warn!("vmxboot: spin rip {rip:#x} is not a physical address to read code from");
        return;
    };
    let virt = match space.direct_map().reach(phys, SPIN_CODE_BYTES) {
        Ok(virt) => virt,
        Err(error) => {
            warn!("vmxboot: could not reach the spin site at {rip:#x}: {error:?}");
            return;
        }
    };
    let len = usize::try_from(SPIN_CODE_BYTES).unwrap_or(0);
    // SAFETY: the direct map reaches `SPIN_CODE_BYTES` bytes from `virt`, which
    // lie in the RAM the guest is executing from; the bytes are only read, and
    // the guest is not running during this exit, so nothing writes them.
    let code = unsafe { core::slice::from_raw_parts(virt.as_u64() as *const u8, len) };
    info!("vmxboot: spin code at {rip:#x}: {code:02x?}");
}

/// Answers one portal notification, stepping over its `VMCALL` and saying
/// whether the guest should be resumed.
///
/// The notification is in the guest's `RDX` and any status it carries in `RAX`,
/// exactly as the portal left them.
fn notified(cell: &Vmcs, registers: &Registers, portal: &Portal) -> bool {
    let marker = registers.rdx;
    let status = registers.rax;
    let keep_going = match Notification::from_bits(marker) {
        Some(Notification::LoaderUnloaded) => loader(portal, true, status),
        Some(Notification::LoaderSkipped) => loader(portal, false, status),
        Some(Notification::ExitSucceeded) => {
            exit_succeeded(portal);
            false
        }
        Some(Notification::StartReturned) => {
            error!(
                "vmxboot: firmware StartImage returned status {status:#x}; no boot manager was preloaded to run"
            );
            false
        }
        None => {
            error!("vmxboot: the guest issued VMCALL with an unknown marker {marker:#x}");
            false
        }
    };
    if !keep_going {
        return false;
    }
    // SAFETY: `cell` is current; the exit was on the portal's VMCALL, whose
    // length the processor recorded, so stepping past it is sound.
    match unsafe { cell.advance_past_instruction() } {
        Ok(()) => true,
        Err(error) => {
            error!("vmxboot: could not step past the portal VMCALL: {error}");
            false
        }
    }
}

/// Handles the first `ExitBootServices` hook: wipes the loader when firmware
/// released its image, records which way it went, and resumes the guest.
fn loader(portal: &Portal, unloaded: bool, status: u64) -> bool {
    let wiped = if unloaded {
        match portal.wipe_loader() {
            Ok(()) => {
                info!("vmxboot: hv-loader image unloaded and wiped");
                true
            }
            Err(error) => {
                warn!("vmxboot: hv-loader was unloaded but could not be wiped: {error}");
                false
            }
        }
    } else {
        warn!("vmxboot: firmware rejected hv-loader UnloadImage with status {status:#x}");
        false
    };
    let transition = if wiped {
        portal.mark_loader_handled()
    } else {
        portal.mark_loader_skipped()
    };
    if let Err(error) = transition {
        error!("vmxboot: could not publish the loader state: {error}");
        return false;
    }
    true
}

/// Handles firmware's successful `ExitBootServices`: restores the boot-services
/// table and reports that the VMX path past firmware is not built yet.
fn exit_succeeded(portal: &Portal) {
    match portal.restore_boot_services() {
        Ok(()) => {
            info!("vmxboot: firmware ExitBootServices succeeded; boot-services table restored");
        }
        Err(error) => error!("vmxboot: could not restore the boot-services table: {error}"),
    }
    error!(
        "vmxboot: firmware's services are gone, but the VMX path past ExitBootServices (starting the other processors, concealing the portal, interposing on devices) is not built yet; halting"
    );
}

/// Byte offsets of the local-APIC registers in the register page, which the
/// virtual-APIC page lays out exactly as the memory-mapped controller does.
const APIC_ID: usize = 0x20;
/// Offset of the version register.
const APIC_VERSION: usize = 0x30;
/// Offset of the task-priority register, which the TPR shadow also uses.
const APIC_TASK_PRIORITY: usize = 0x80;
/// Offset of the processor-priority register.
const APIC_PROCESSOR_PRIORITY: usize = 0xA0;
/// Offset of the logical-destination register.
const APIC_LOGICAL_DESTINATION: usize = 0xD0;
/// Offset of the destination-format register.
const APIC_DESTINATION_FORMAT: usize = 0xE0;
/// Offset of the spurious-interrupt-vector register.
const APIC_SPURIOUS: usize = 0xF0;
/// Offset of the first in-service-register word.
const APIC_IN_SERVICE: usize = 0x100;
/// Offset of the first trigger-mode-register word.
const APIC_TRIGGER_MODE: usize = 0x180;
/// Offset of the first interrupt-request-register word.
const APIC_INTERRUPT_REQUEST: usize = 0x200;
/// Offset of the error-status register.
const APIC_ERROR_STATUS: usize = 0x280;
/// Offset of the corrected-machine-check local-vector-table entry.
const APIC_LVT_CMCI: usize = 0x2F0;
/// Offset of the low half of the interrupt-command register.
const APIC_COMMAND_LOW: usize = 0x300;
/// Offset of the high half of the interrupt-command register.
const APIC_COMMAND_HIGH: usize = 0x310;
/// Offset of the timer local-vector-table entry.
const APIC_LVT_TIMER: usize = 0x320;
/// Offset of the thermal-sensor local-vector-table entry.
const APIC_LVT_THERMAL: usize = 0x330;
/// Offset of the performance-counter local-vector-table entry.
const APIC_LVT_PERFORMANCE: usize = 0x340;
/// Offset of the first interrupt-pin local-vector-table entry.
const APIC_LVT_LINT0: usize = 0x350;
/// Offset of the second interrupt-pin local-vector-table entry.
const APIC_LVT_LINT1: usize = 0x360;
/// Offset of the error local-vector-table entry.
const APIC_LVT_ERROR: usize = 0x370;
/// Offset of the timer's initial-count register.
const APIC_TIMER_INITIAL_COUNT: usize = 0x380;
/// Offset of the timer's current-count register.
const APIC_TIMER_CURRENT_COUNT: usize = 0x390;
/// Offset of the timer's divide-configuration register.
const APIC_TIMER_DIVIDE: usize = 0x3E0;
/// Bytes between one 32-bit register word and the next in a bank.
const APIC_REGISTER_STRIDE: usize = 0x10;

/// The two 32-bit halves of a 64-bit value, low then high, without a narrowing
/// cast.
fn halves(value: u64) -> [u32; 2] {
    let bytes = value.to_le_bytes();
    [
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
    ]
}

/// Seeds a virtual-APIC page with the controller state firmware was captured
/// with, so the guest reads back what it last left in each register rather than
/// the zeros of a fresh page.
///
/// Each value goes to the offset the memory-mapped controller holds it at,
/// which is the layout the virtual-APIC page shares. The interrupt-command
/// register is the one that spans two registers; everything else is a single
/// 32-bit word, and the request, in-service and trigger-mode banks are eight
/// words apart by the register stride.
fn seed_vapic(page: &mut Page, local: &LocalState) {
    let mut put = |offset: usize, value: u32| {
        page.0[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    put(APIC_ID, local.id);
    put(APIC_VERSION, local.version);
    put(APIC_TASK_PRIORITY, local.task_priority);
    put(APIC_PROCESSOR_PRIORITY, local.processor_priority);
    put(APIC_LOGICAL_DESTINATION, local.logical_destination);
    put(APIC_DESTINATION_FORMAT, local.destination_format);
    put(APIC_SPURIOUS, local.spurious);
    put(APIC_ERROR_STATUS, local.error_status);
    put(APIC_LVT_CMCI, local.lvt_corrected_machine_check);
    let [command_low, command_high] = halves(local.command);
    put(APIC_COMMAND_LOW, command_low);
    put(APIC_COMMAND_HIGH, command_high);
    put(APIC_LVT_TIMER, local.lvt_timer);
    put(APIC_LVT_THERMAL, local.lvt_thermal);
    put(APIC_LVT_PERFORMANCE, local.lvt_performance);
    put(APIC_LVT_LINT0, local.lvt_lint0);
    put(APIC_LVT_LINT1, local.lvt_lint1);
    put(APIC_LVT_ERROR, local.lvt_error);
    put(APIC_TIMER_INITIAL_COUNT, local.timer_initial_count);
    put(APIC_TIMER_CURRENT_COUNT, local.timer_current_count);
    put(APIC_TIMER_DIVIDE, local.timer_divide);
    for (index, word) in local.in_service.iter().enumerate() {
        put(APIC_IN_SERVICE + index * APIC_REGISTER_STRIDE, *word);
    }
    for (index, word) in local.trigger_mode.iter().enumerate() {
        put(APIC_TRIGGER_MODE + index * APIC_REGISTER_STRIDE, *word);
    }
    for (index, word) in local.interrupt_request.iter().enumerate() {
        put(APIC_INTERRUPT_REQUEST + index * APIC_REGISTER_STRIDE, *word);
    }
}

/// Virtualizes the guest's local APIC: seeds a virtual-APIC page from the
/// captured controller, remaps the guest's APIC page to a 4-KiB access page the
/// processor watches, and turns the APIC-virtualization controls on.
///
/// Returns the two pages, which the caller keeps alive for the run because the
/// processor reads the virtual-APIC page throughout it. On any failure it logs
/// and returns `None`, leaving the guest to reach the real controller — which
/// is worse but is the state the probe was in before.
fn setup_apic(
    space: &AddressSpace,
    firmware: &FirmwareContext,
    memory: &mut EptFrames,
    eptp_root: u64,
    cell: &Vmcs,
) -> Option<(Box<Page>, Box<Page>)> {
    // SAFETY: this runs in VMX operation, so the capability register exists.
    if !unsafe { controls::apic_virtualization_available() } {
        warn!(
            "vmxboot: this processor cannot virtualize the APIC; the guest would reach the real one"
        );
        return None;
    }
    // x2APIC reaches the controller through model-specific registers rather than
    // the page this virtualizes, so warn where firmware left it in that mode:
    // the MMIO virtualization below does not cover it.
    if firmware.interrupts.base & X2APIC_ENABLE != 0 {
        warn!(
            "vmxboot: firmware left the APIC in x2APIC mode; the guest reaches it through MSRs, which this virtualization does not yet cover"
        );
    }
    let mut vapic = Page::zeroed();
    let access = Page::zeroed();
    seed_vapic(&mut vapic, &firmware.interrupts.local);
    let at = |page: &Page| VirtAddr::new(core::ptr::from_ref(&page.0).addr() as u64);
    let (Ok(vapic_phys), Ok(access_phys)) =
        (space.translate(at(&vapic)), space.translate(at(&access)))
    else {
        error!("vmxboot: could not translate the APIC virtualization pages");
        return None;
    };
    let apic_page = firmware.interrupts.base & !(PAGE_BYTES as u64 - 1);
    if let Err(error) = ept::map(
        memory,
        eptp_root,
        apic_page,
        access_phys.as_u64(),
        EptAccess::READ | EptAccess::WRITE,
        EptMemoryType::Uncacheable,
    ) {
        error!("vmxboot: could not map the guest APIC page to the access page: {error:?}");
        return None;
    }
    // SAFETY: `cell` is the current VMCS in VMX operation, and `program` wrote
    // the controls this adds to.
    let enabled =
        unsafe { controls::virtualize_apic(cell, vapic_phys.as_u64(), access_phys.as_u64()) };
    if let Err(error) = enabled {
        error!("vmxboot: could not enable APIC virtualization: {error}");
        return None;
    }
    info!(
        "vmxboot: guest APIC virtualized; accesses to {apic_page:#x} reach the virtual-APIC page, not the real controller"
    );
    Some((vapic, access))
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
        cell.write(Field::GUEST_RFLAGS, save.rflags)?;
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
        Exit::Stopped(Stop::Unhandled(BasicExitReason::INIT_SIGNAL)) => error!(
            "vmxboot: the firmware guest took an INIT signal: an INIT inter-processor interrupt reached this processor, which is how a processor is reset. The guest drove the real local APIC through the identity EPT, so the interrupt controller is not virtualized and the platform resets rather than citrine handling it; a virtual local APIC is what this needs."
        ),
        Exit::Stopped(Stop::Unhandled(BasicExitReason::STARTUP_IPI)) => error!(
            "vmxboot: the firmware guest took a start-up IPI: the guest drove the real local APIC through the identity EPT to start a processor, which the host does not yet virtualize."
        ),
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
