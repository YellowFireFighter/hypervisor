//! A one-shot check that this processor can enter VMX operation and that the
//! VMCS instruction wrappers round-trip.
//!
//! The Intel path in `vmx` and `vmcs` is written but unverified: nothing has
//! ever executed a VMX instruction. This runs the first piece of it that needs
//! no guest and no world switch — enter VMX operation, make a VMCS current,
//! `VMWRITE` a field and `VMREAD` it back — on a real processor, and reports
//! whether it worked. It is built only behind the `vmx-selftest` feature, runs
//! early in bring-up and is followed by a halt, because the rest of pulzar is
//! AMD SVM and cannot run on the Intel machine this is meant for.
//!
//! Everything here is reported through [`log`], so a build that can show it is
//! one with a logging backend the tester can read — a serial port, the debug
//! console, or the on-screen log (`--screen-log`).

use alloc::{boxed::Box, vec::Vec};

use ept::{ENTRIES, Memory};
use log::{error, info, warn};
use paging::AddressSpace;
use vmcs::{Entered, Registers, VmFail, Vmcs, controls, guest, host, instr, run};
use vmexits::Flow;
use vmx::{
    BasicExitReason, EptAccess, EptEntry, EptMemoryType, EptPointer, ExitReason, Field,
    Interruption, PAGE_BYTES, VmxBasic, event::Kind,
};
use x86_64::{
    VirtAddr,
    registers::segmentation::{CS, Segment},
};

/// How many GiB of guest-physical memory the EPT guest identity-maps, which
/// must cover every host-physical address its code, stack and page tables fall
/// at.
const EPT_IDENTITY_GIB: usize = 16;

/// A page-aligned, page-sized region, which is what a VMXON region and a VMCS
/// each are.
#[repr(C, align(4096))]
struct Page([u8; PAGE_BYTES]);

impl Page {
    /// A fresh zeroed page on the heap, which lives in the chunk the direct map
    /// covers so its physical address is reachable.
    fn zeroed() -> Box<Self> {
        Box::new(Self([0; PAGE_BYTES]))
    }
}

/// Runs the self-test, logging each step and a final `PASS` or `FAIL`.
///
/// `space` is the live address space, used only to translate the two heap pages
/// to the physical addresses the VMX instructions take. Any failure is logged
/// and returns; nothing here panics or stops the caller.
///
/// Always returns `true`, which the caller uses to guard its halting return so
/// the compiler does not flag the SVM path below it unreachable. The result is
/// a signal, not a verdict — `PASS` or `FAIL` is in the log, not the bool.
#[must_use]
pub(crate) fn run(space: &AddressSpace) -> bool {
    info!("vmx: self-test starting");

    let mut vmxon_region = Page::zeroed();
    let mut vmcs_region = Page::zeroed();
    let vmxon_virt = VirtAddr::new(core::ptr::from_ref(&vmxon_region.0).addr() as u64);
    let vmcs_virt = VirtAddr::new(core::ptr::from_ref(&vmcs_region.0).addr() as u64);

    let (Ok(vmxon_phys), Ok(vmcs_phys)) = (space.translate(vmxon_virt), space.translate(vmcs_virt))
    else {
        error!("vmx: self-test could not translate its pages to physical addresses");
        return true;
    };
    info!("vmx: vmxon region at {vmxon_phys:#x}, vmcs at {vmcs_phys:#x}");

    // SAFETY: the two pages are freshly allocated, exclusively owned here, and
    // their physical addresses were just translated from their own virtual
    // ones; this runs in long mode with paging on. `vmxon_region` lives until
    // the end of this function, so the region stays valid for the whole test.
    let vmx = match unsafe { vmcs::enter(vmxon_region.0.as_mut_ptr().cast(), vmxon_phys) } {
        Ok(vmx) => vmx,
        Err(error) => {
            error!("vmx: SELF-TEST FAIL: could not enter VMX operation: {error}");
            return true;
        }
    };
    info!(
        "vmx: entered VMX operation; VMCS revision {:#x}",
        vmx.basic().revision()
    );

    // SAFETY: this processor is in VMX operation, the VMCS page is exclusively
    // owned, and `vmx.basic()` is this processor's own IA32_VMX_BASIC so the
    // revision matches.
    let mut cell = match unsafe {
        Vmcs::activate(vmcs_region.0.as_mut_ptr().cast(), vmcs_phys, vmx.basic())
    } {
        Ok(cell) => cell,
        Err(error) => {
            error!("vmx: SELF-TEST FAIL: VMCLEAR/VMPTRLD: {error}");
            // SAFETY: in VMX operation with no current VMCS (activate failed).
            unsafe { leave() };
            return true;
        }
    };

    battery(&mut cell, space, vmx.basic());

    // SAFETY: `cell` is current, so clearing it leaves no current VMCS, which is
    // the precondition for leaving VMX operation.
    unsafe {
        let _ = instr::vmclear(cell.region()).ok();
        leave();
    }
    info!("vmx: self-test complete");
    true
}

/// Runs every check against the current VMCS `cell`, logging each `[PASS]` or
/// `[FAIL]` and a final tally, so one boot exercises many things rather than
/// one. `basic` is this processor's `IA32_VMX_BASIC`, read while entering.
fn battery(cell: &mut Vmcs, space: &AddressSpace, basic: VmxBasic) {
    let mut passed = 0_u32;
    let mut total = 0_u32;
    let mut check = |name: &str, ok: bool| {
        total += 1;
        if ok {
            passed += 1;
            info!("vmx: [PASS] {name}");
        } else {
            error!("vmx: [FAIL] {name}");
        }
    };

    // The enable path's own results, read back from hardware: it must have set
    // CR4.VMXE, and IA32_VMX_BASIC must describe a VMCS that fits a page.
    check("CR4.VMXE set by the enable path", cr4_vmxe_set());
    check(
        "VMX basic region fits a page",
        (1..=PAGE_BYTES).contains(&(basic.region_bytes() as usize)),
    );

    // Field round-trips across every width, which exercises the encoding of
    // each: a wrong width or index would read back something other than what
    // was written.
    check(
        "16-bit field round-trip",
        roundtrip(cell, Field::GUEST_ES_SELECTOR, 0x1234),
    );
    check(
        "32-bit field round-trip",
        roundtrip(cell, Field::GUEST_ES_LIMIT, 0xDEAD_BEEF),
    );
    check(
        "64-bit field round-trip",
        roundtrip(cell, Field::TSC_OFFSET, 0x1122_3344_5566_7788),
    );
    check(
        "natural-width field round-trip",
        roundtrip(cell, Field::GUEST_RIP, 0x0000_0000_0040_1000),
    );
    check(
        "guest launch, resume across CPUIDs, VMCALL",
        entry_probe(cell),
    );
    check("guest GPRs saved on exit", register_save_probe(cell));
    // SAFETY: we reached here only by entering VMX operation, so this is a
    // VMX-capable processor and the capability registers exist.
    if unsafe { controls::ept_available() } {
        check(
            "guest launch under EPT (second translation)",
            ept_probe(cell, space),
        );
        check(
            "guest runs in its own address space behind a non-identity EPT",
            own_address_space_probe(cell, space),
        );
    } else {
        warn!("vmx: EPT not available on this processor; skipping the EPT guest checks");
    }
    check(
        "second VMCS switch and independence",
        second_vmcs_probe(cell, space, basic),
    );
    check(
        "CPUID emulated through the dispatch loop",
        cpuid_dispatch_probe(cell),
    );
    check(
        "RDMSR emulated through the dispatch loop",
        rdmsr_dispatch_probe(cell),
    );
    check(
        "WRMSR forwarded through the dispatch loop",
        wrmsr_dispatch_probe(cell),
    );
    check(
        "CPUID conceals the virtualization extension",
        cpuid_conceal_probe(cell),
    );
    check(
        "IA32_FEATURE_CONTROL answered as firmware-locked",
        feature_control_probe(cell),
    );
    check(
        "event injected through the guest IDT",
        inject_probe(cell, 6, false),
    );
    check(
        "event with error code injected through the guest IDT",
        inject_probe(cell, 13, true),
    );

    info!("vmx: SELF-TEST SUMMARY: {passed}/{total} checks passed");
}

/// A guest that exits three times: two `CPUID`s, each an unconditional VM exit,
/// then a `VMCALL` that ends the probe.
///
/// This exercises not just entry but resumption: the host advances past each
/// `CPUID` and re-enters, so the second `CPUID` and the `VMCALL` are reached
/// only if `VMRESUME` and the instruction-length advance both work. It is a
/// naked function so its address is in the image's executable text, which the
/// guest reaches through the host's own page tables; the `HLT` loop is never
/// reached.
#[unsafe(naked)]
unsafe extern "C" fn guest_probe() {
    core::arch::naked_asm!("cpuid", "cpuid", "vmcall", "2:", "hlt", "jmp 2b");
}

/// The most guest entries the probe makes before giving up, so a guest that
/// never reaches its `VMCALL` cannot spin the host.
const MAX_ENTRIES: u32 = 8;

/// Programs the flat 64-bit guest into `cell`, behind `ept` when given.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation, and the
/// `rip`/`rsp` must point at host-executable code and host-writable stack.
unsafe fn program_guest(
    cell: &Vmcs,
    ept: Option<EptPointer>,
    rip: u64,
    rsp: u64,
) -> Result<(), VmFail> {
    // SAFETY: the caller guarantees the current VMCS and valid rip/rsp.
    unsafe {
        host::program(cell)
            .and_then(|()| controls::program(cell, ept))
            .and_then(|()| guest::program(cell, rip, rsp))
    }
}

/// Drives the guest `cell` describes through its `CPUID` exits until it reaches
/// a `VMCALL`, returning whether it did.
///
/// Each `CPUID` is resumed past with the recorded instruction length, so
/// reaching the `VMCALL` means launch, resume and the RIP advance all work.
fn drive_to_vmcall(cell: &mut Vmcs) -> bool {
    let mut registers = Registers::default();
    let mut cpuids = 0_u32;
    for entry in 1..=MAX_ENTRIES {
        // SAFETY: `cell` is current and fully programmed, and the run loop
        // preserves and restores the host around each entry.
        match unsafe { run::run(cell, &mut registers) } {
            Entered::Failed(fail) => {
                // SAFETY: `cell` is still current.
                let number = unsafe { cell.read(Field::VM_INSTRUCTION_ERROR) }.unwrap_or(0);
                error!("vmx: entry {entry} rejected ({fail}); VM-instruction-error {number}");
                return false;
            }
            Entered::Exited => {
                let Some(reason) = exit_reason(cell) else {
                    error!("vmx: guest exited but EXIT_REASON could not be read");
                    return false;
                };
                if reason == BasicExitReason::CPUID {
                    cpuids += 1;
                    // SAFETY: `cell` is current and the exit was on CPUID, whose
                    // length the processor recorded.
                    if let Err(error) = unsafe { cell.advance_past_instruction() } {
                        error!("vmx: could not advance past CPUID: {error}");
                        return false;
                    }
                } else if reason == BasicExitReason::VMCALL {
                    info!("vmx: guest ran {cpuids} CPUIDs then VMCALL over {entry} entries");
                    return true;
                } else {
                    error!(
                        "vmx: guest exited with reason {} (expected CPUID = 10 or VMCALL = 18)",
                        reason.number()
                    );
                    return false;
                }
            }
        }
    }
    warn!("vmx: gave up after {MAX_ENTRIES} entries without a VMCALL");
    false
}

/// Launches the flat 64-bit guest in the host's own address space and drives it
/// to its `VMCALL`.
fn entry_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: guest-entry VMCS programming failed: {error}");
        return false;
    }
    let outcome = drive_to_vmcall(cell);
    drop(stack);
    outcome
}

/// Launches the same guest behind an identity EPT, so it runs through a second
/// translation rather than directly in the host's physical layout.
///
/// Reaching the `VMCALL` without an EPT violation or misconfiguration means the
/// EPT was built right, installed, and walked by the processor.
fn ept_probe(cell: &mut Vmcs, space: &AddressSpace) -> bool {
    let mut memory = EptMemory {
        frames: Vec::new(),
        space,
    };
    let eptp = match ept::identity(&mut memory, EPT_IDENTITY_GIB) {
        Ok(pointer) => pointer,
        Err(error) => {
            error!("vmx: EPT could not be built: {error:?}");
            return false;
        }
    };

    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; the EPT
    // identity-maps the host-physical addresses the guest reaches, so `rip` and
    // `rsp`, which the guest translates through the host's own CR3 to those
    // addresses, stay reachable.
    if let Err(error) = unsafe { program_guest(cell, Some(eptp), rip, rsp) } {
        error!("vmx: EPT-guest VMCS programming failed: {error}");
        return false;
    }
    let outcome = drive_to_vmcall(cell);
    // The EPT tables and stack are walked by the processor for the whole of the
    // run above, so they are dropped only now.
    drop(stack);
    drop(memory);
    outcome
}

/// EPT table frames for the self-test, allocated from the heap and translated
/// to the physical addresses the processor walks.
struct EptMemory<'a> {
    frames: Vec<(Box<Page>, u64)>,
    space: &'a AddressSpace,
}

impl Memory for EptMemory<'_> {
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
        // SAFETY: a `Page` is 4-KiB aligned and 4-KiB long, exactly a table of
        // `ENTRIES` eight-byte entries, so the pointer is suitably aligned; it
        // is owned here and reached through a unique borrow.
        unsafe { &mut *page.cast::<[EptEntry; ENTRIES]>() }
    }
}

/// The value the register-save guest writes into `RBX` before it exits, chosen
/// to be recognizable and to span all eight bytes.
const REGISTER_MARKER: u64 = 0x1122_3344_5566_7788;

/// A guest that writes a marker into `RBX` and then `VMCALL`s.
///
/// Unlike `CPUID`, the `MOV` runs before the exit, so the marker is in the
/// guest's `RBX` when it exits — which is what lets the host check the world
/// switch saved the guest's registers rather than only entering and leaving.
#[unsafe(naked)]
unsafe extern "C" fn guest_marks() {
    core::arch::naked_asm!("mov rbx, {marker}", "vmcall", "2:", "hlt", "jmp 2b", marker = const REGISTER_MARKER);
}

/// Launches the marking guest and checks the marker it wrote came back in the
/// register block.
fn register_save_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_marks as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is current and VMX operation is live; the guest runs in the
    // host's address space.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: register-save programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    // SAFETY: `cell` is current and programmed.
    let entered = unsafe { run::run(cell, &mut registers) };
    let ok = entered == Entered::Exited
        && exit_reason(cell) == Some(BasicExitReason::VMCALL)
        && registers.rbx == REGISTER_MARKER;
    if !ok {
        error!(
            "vmx: register-save: {entered:?}, reason {:?}, rbx {:#x}",
            exit_reason(cell),
            registers.rbx
        );
    }
    drop(stack);
    ok
}

/// Makes a second VMCS current, round-trips a field in it, then restores the
/// first — exercising VMCS switching and that each VMCS is independent.
fn second_vmcs_probe(cell: &Vmcs, space: &AddressSpace, basic: VmxBasic) -> bool {
    let mut page = Page::zeroed();
    let virt = VirtAddr::new(core::ptr::from_ref(&page.0).addr() as u64);
    let Ok(phys) = space.translate(virt) else {
        error!("vmx: second VMCS page could not be translated");
        return false;
    };

    // SAFETY: VMX operation is live, the page is an exclusively-owned,
    // page-sized VMCS region, and `basic` is this processor's own.
    let second = match unsafe { Vmcs::activate(page.0.as_mut_ptr().cast(), phys, basic) } {
        Ok(second) => second,
        Err(error) => {
            error!("vmx: second VMCS activate failed: {error}");
            return false;
        }
    };
    // The second VMCS is current now; a field written here must read back from
    // it, independent of the first.
    let ok = roundtrip(&second, Field::GUEST_RSP, 0xCAFE_0000_1000);

    // SAFETY: `second` is current; clearing it and reloading the first leaves
    // the first current again for the caller's cleanup.
    unsafe {
        let _ = instr::vmclear(second.region()).ok();
        let _ = instr::vmptrld(cell.region()).ok();
    }
    drop(page);
    ok
}

/// Drives the guest `cell` describes through the real [`vmexits::dispatch`]
/// until it makes a `VMCALL`, leaving the guest's registers in `registers`.
///
/// Unlike [`drive_to_vmcall`], which advances past each exit itself, this hands
/// every exit to the dispatch loop a running hypervisor would use. Reaching the
/// `VMCALL` means that loop emulated each exit the guest took — placing a
/// `CPUID` or `RDMSR` result in the guest's registers — and resumed it
/// correctly, which is what distinguishes it from a loop that only steps over
/// the instruction.
fn drive_dispatch(cell: &mut Vmcs, registers: &mut Registers) -> bool {
    for entry in 1..=MAX_ENTRIES {
        // SAFETY: `cell` is current and fully programmed, and the run loop
        // preserves and restores the host around each entry.
        match unsafe { run::run(cell, registers) } {
            Entered::Failed(fail) => {
                // SAFETY: `cell` is still current.
                let number = unsafe { cell.read(Field::VM_INSTRUCTION_ERROR) }.unwrap_or(0);
                error!("vmx: entry {entry} rejected ({fail}); VM-instruction-error {number}");
                return false;
            }
            Entered::Exited => {
                // SAFETY: `cell` is current and `registers` is the block the run
                // loop just filled; `probe::install` claimed the
                // general-protection vector during bring-up.
                match unsafe { vmexits::dispatch(cell, registers) } {
                    Flow::Resume => {}
                    Flow::Vmcall => return true,
                    Flow::Stop(stop) => {
                        error!("vmx: dispatch stopped the guest: {stop:?}");
                        return false;
                    }
                }
            }
        }
    }
    warn!("vmx: dispatch gave up after {MAX_ENTRIES} entries without a VMCALL");
    false
}

/// A guest that runs `CPUID` leaf 0 and stashes the three vendor words it gets
/// back, then `VMCALL`s.
///
/// Leaf 0 returns the processor's vendor string in `EBX`, `ECX` and `EDX`,
/// which the host can read for itself. Moving them into registers that survive
/// to the `VMCALL` lets the host check the dispatch loop placed the real result
/// in the guest rather than only advancing past the instruction.
#[unsafe(naked)]
unsafe extern "C" fn guest_cpuid() {
    core::arch::naked_asm!(
        "xor eax, eax",
        "cpuid",
        "mov r8, rbx",
        "mov r9, rcx",
        "mov r10, rdx",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b"
    );
}

/// Runs the CPUID guest through the dispatch loop and checks the vendor words
/// it received match the machine's own, which proves the loop emulated the exit
/// and resumed the guest.
fn cpuid_dispatch_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_cpuid as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: CPUID-dispatch programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let expected = processor::cpuid(0, 0);
    let ok = reached
        && registers.r8 == u64::from(expected.ebx)
        && registers.r9 == u64::from(expected.ecx)
        && registers.r10 == u64::from(expected.edx);
    if !ok {
        error!(
            "vmx: CPUID dispatch: reached {reached}, ebx {:#x} vs {:#x}",
            registers.r8, expected.ebx
        );
    }
    drop(stack);
    ok
}

/// `IA32_EFER`, which every long-mode processor implements, so the guest's
/// `RDMSR` of it is forwarded rather than refused.
const IA32_EFER: u32 = 0xC000_0080;

/// A guest that reads `IA32_EFER` and assembles the two halves `RDMSR` returns
/// into one register, then `VMCALL`s.
///
/// `RDMSR` returns the register in `EDX:EAX`; combining them is ordinary guest
/// code that takes no exit, so the register it ends with is the value the
/// dispatch loop forwarded from the machine.
#[unsafe(naked)]
unsafe extern "C" fn guest_rdmsr() {
    core::arch::naked_asm!(
        "mov ecx, {efer}",
        "rdmsr",
        "shl rdx, 32",
        "or rax, rdx",
        "mov r8, rax",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b",
        efer = const IA32_EFER,
    );
}

/// Runs the RDMSR guest through the dispatch loop and checks the value it read
/// matches the machine's own `IA32_EFER`.
fn rdmsr_dispatch_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_rdmsr as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: RDMSR-dispatch programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let expected = probe::read(IA32_EFER).unwrap_or(0);
    let ok = reached && registers.r8 == expected;
    if !ok {
        error!(
            "vmx: RDMSR dispatch: reached {reached}, efer {:#x} vs {expected:#x}",
            registers.r8
        );
    }
    drop(stack);
    ok
}

/// `IA32_KERNEL_GS_BASE`, which every long-mode processor implements and holds
/// an arbitrary canonical value, so a `WRMSR`/`RDMSR` round-trip through it
/// proves the forwarding without depending on an optional register.
const IA32_KERNEL_GS_BASE: u32 = 0xC000_0102;
/// The low half of the value the WRMSR guest round-trips.
const WRMSR_MARKER_LOW: u32 = 0x1122_3344;
/// The high half of that value. Kept below bit 47 so the whole is canonical and
/// `WRMSR` accepts it.
const WRMSR_MARKER_HIGH: u32 = 0x0000_7FFF;

/// A guest that writes a marker into `IA32_KERNEL_GS_BASE` with `WRMSR` and
/// reads it back with `RDMSR`, then `VMCALL`s with what it read.
///
/// Both accesses exit and are forwarded by the dispatch loop, so the value that
/// comes back having survived the round trip is proof the `WRMSR` path wrote
/// the machine's register and the `RDMSR` path read it.
#[unsafe(naked)]
unsafe extern "C" fn guest_wrmsr() {
    core::arch::naked_asm!(
        "mov ecx, {msr}",
        "mov eax, {low}",
        "mov edx, {high}",
        "wrmsr",
        "mov ecx, {msr}",
        "rdmsr",
        "shl rdx, 32",
        "or rax, rdx",
        "mov r8, rax",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b",
        msr = const IA32_KERNEL_GS_BASE,
        low = const WRMSR_MARKER_LOW,
        high = const WRMSR_MARKER_HIGH,
    );
}

/// Runs the WRMSR guest through the dispatch loop and checks the value read
/// back matches the one written.
fn wrmsr_dispatch_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_wrmsr as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: WRMSR-dispatch programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let expected = (u64::from(WRMSR_MARKER_HIGH) << 32) | u64::from(WRMSR_MARKER_LOW);
    let ok = reached && registers.r8 == expected;
    if !ok {
        error!(
            "vmx: WRMSR dispatch: reached {reached}, read {:#x} vs {expected:#x}",
            registers.r8
        );
    }
    drop(stack);
    ok
}

/// `CPUID.01H:ECX[5]`, the VMX bit, and `[31]`, the hypervisor-present bit —
/// the two the concealment clears.
const CPUID_VMX_AND_HYPERVISOR: u64 = (1 << 5) | (1 << 31);

/// A guest that reads the standard feature leaf and the first hypervisor leaf,
/// stashing the feature `ECX` and the OR of every hypervisor-leaf word, then
/// `VMCALL`s.
///
/// The feature `ECX` should have the virtualization and hypervisor-present bits
/// clear, and the hypervisor leaf should read as all zero — which is what the
/// concealment in the dispatch loop makes of them.
#[unsafe(naked)]
unsafe extern "C" fn guest_cpuid_conceal() {
    core::arch::naked_asm!(
        "mov eax, 1",
        "cpuid",
        "mov r8, rcx",
        "mov eax, 0x40000000",
        "cpuid",
        "mov r9, rax",
        "or r9, rbx",
        "or r9, rcx",
        "or r9, rdx",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b"
    );
}

/// Runs the concealment guest through the dispatch loop and checks the
/// virtualization extension is hidden: the two feature bits clear and the
/// hypervisor leaf empty.
fn cpuid_conceal_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_cpuid_conceal as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: CPUID-conceal programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let ok = reached && registers.r8 & CPUID_VMX_AND_HYPERVISOR == 0 && registers.r9 == 0;
    if !ok {
        error!(
            "vmx: CPUID conceal: reached {reached}, feature ecx {:#x}, hv-leaf or {:#x}",
            registers.r8, registers.r9
        );
    }
    drop(stack);
    ok
}

/// `IA32_FEATURE_CONTROL`, which the dispatch answers as firmware-locked.
const IA32_FEATURE_CONTROL: u32 = 0x3A;
/// Bit 0, the lock, which the concealed value sets.
const FEATURE_CONTROL_LOCK: u64 = 1 << 0;
/// Bits 1 and 2, the two `VMXON` permissions, which the concealed value clears.
const FEATURE_CONTROL_VMXON: u64 = (1 << 1) | (1 << 2);

/// A guest that reads `IA32_FEATURE_CONTROL` and `VMCALL`s with it.
#[unsafe(naked)]
unsafe extern "C" fn guest_feature_control() {
    core::arch::naked_asm!(
        "mov ecx, {msr}",
        "rdmsr",
        "shl rdx, 32",
        "or rax, rdx",
        "mov r8, rax",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b",
        msr = const IA32_FEATURE_CONTROL,
    );
}

/// Runs the feature-control guest through the dispatch loop and checks the
/// register is answered as firmware-locked: the lock set, both `VMXON`
/// permissions clear.
fn feature_control_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_feature_control as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS and VMX operation is live; `rip` is in
    // executable image text and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe { program_guest(cell, None, rip, rsp) } {
        error!("vmx: feature-control programming failed: {error}");
        drop(stack);
        return false;
    }
    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let ok = reached
        && registers.r8 & FEATURE_CONTROL_LOCK != 0
        && registers.r8 & FEATURE_CONTROL_VMXON == 0;
    if !ok {
        error!(
            "vmx: feature control: reached {reached}, value {:#x}",
            registers.r8
        );
    }
    drop(stack);
    ok
}

/// Guest-physical — and, through the guest's identity paging, guest-virtual —
/// address of the data page the own-address-space guest reads and writes.
const GUEST_DATA: u64 = 0x2000;
/// The top of the own-address-space guest's stack; its stack page is the 4 KiB
/// below this.
const GUEST_STACK_TOP: u64 = 0x4000;
/// Guest-virtual base the own-address-space guest's code runs at.
const GUEST_CODE: u64 = 0x1_0000;
/// Guest-physical addresses of the guest's own paging structures, which the
/// processor reaches through EPT from the guest's `CR3`, never through a
/// guest-virtual address.
const GUEST_PML4: u64 = 0x2_0000;
/// Guest-physical address of the guest page-directory-pointer table.
const GUEST_PDPT: u64 = 0x2_1000;
/// Guest-physical address of the guest page directory.
const GUEST_PD: u64 = 0x2_2000;

/// The value the host plants in the guest's data page for it to read.
const GUEST_PLANTED: u64 = 0xFEED_FACE_CAFE_BEEF;
/// The value the guest writes into its data page and reads back, which proves
/// the EPT mapping of that page is writable.
const GUEST_WRITTEN: u64 = 0x0BAD_C0DE_1234_5678;

/// Present and writable: the flags a guest paging entry pointing at a lower
/// level carries.
const PTE_PRESENT_WRITABLE: u64 = 0b11;
/// The page-size bit, which marks a page-directory entry a 2-MiB leaf.
const PTE_LARGE: u64 = 1 << 7;

/// The access every page of the own-address-space guest is granted in EPT:
/// read, write and execute, so one mapping serves code, stack, data and the
/// guest's own page tables alike.
const GUEST_EPT_ACCESS: EptAccess = EptAccess::READ
    .union(EptAccess::WRITE)
    .union(EptAccess::EXECUTE);

/// A guest that reads a planted value from its own data page, writes a second
/// value there and reads it back, then `VMCALL`s with both.
///
/// It runs on its own page tables, so the address `0x2000` it names is a
/// guest-virtual address its paging turns into a guest-physical one, which EPT
/// then turns into the host frame the host planted the value in — two
/// translations, neither of them the identity the earlier EPT guest used.
#[unsafe(naked)]
unsafe extern "C" fn guest_own_space() {
    core::arch::naked_asm!(
        "mov rdi, {data}",
        "mov r8, [rdi]",
        "mov rax, {written}",
        "mov [rdi], rax",
        "mov r9, [rdi]",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b",
        data = const GUEST_DATA,
        written = const GUEST_WRITTEN,
    );
}

/// Runs a guest in its own address space behind a non-identity EPT and checks
/// it read the planted value and the value it wrote back.
///
/// The guest's three live pages (code, data, stack) and its three paging
/// structures are each given a guest-physical address of the host's choosing
/// and mapped through EPT to a host frame that is not that address, so reaching
/// the `VMCALL` with both values proves the guest paging and the EPT
/// translation both work and compose.
fn own_address_space_probe(cell: &mut Vmcs, space: &AddressSpace) -> bool {
    let mut data = Page::zeroed();
    let stack = Page::zeroed();
    let mut pml4 = Page::zeroed();
    let mut pdpt = Page::zeroed();
    let mut pd = Page::zeroed();

    data.0[..8].copy_from_slice(&GUEST_PLANTED.to_le_bytes());
    pml4.0[..8].copy_from_slice(&(GUEST_PDPT | PTE_PRESENT_WRITABLE).to_le_bytes());
    pdpt.0[..8].copy_from_slice(&(GUEST_PD | PTE_PRESENT_WRITABLE).to_le_bytes());
    // The one page-directory entry is a 2-MiB leaf mapping guest-virtual
    // 0..2 MiB onto guest-physical 0..2 MiB, which covers every address the
    // guest names.
    pd.0[..8].copy_from_slice(&(PTE_LARGE | PTE_PRESENT_WRITABLE).to_le_bytes());

    let at = |page: &Page| VirtAddr::new(core::ptr::from_ref(&page.0).addr() as u64);
    let (Ok(data_pa), Ok(stack_pa), Ok(pml4_pa), Ok(pdpt_pa), Ok(pd_pa)) = (
        space.translate(at(&data)),
        space.translate(at(&stack)),
        space.translate(at(&pml4)),
        space.translate(at(&pdpt)),
        space.translate(at(&pd)),
    ) else {
        error!("vmx: own-address could not translate its frames");
        return false;
    };

    // The guest's code runs in place: its real host page is mapped into the
    // guest at GUEST_CODE, so no bytes are copied. A second page follows in case
    // the function straddles a page boundary.
    let fn_va = (guest_own_space as *const ()).addr() as u64;
    let code_page = fn_va & !0xFFF;
    let (Ok(code_pa0), Ok(code_pa1)) = (
        space.translate(VirtAddr::new(code_page)),
        space.translate(VirtAddr::new(code_page + PAGE_BYTES as u64)),
    ) else {
        error!("vmx: own-address could not translate its code page");
        return false;
    };

    let mut memory = EptMemory {
        frames: Vec::new(),
        space,
    };
    let Some(root) = memory.allocate() else {
        error!("vmx: own-address could not allocate the EPT root");
        return false;
    };
    let mappings = [
        (GUEST_DATA, data_pa.as_u64()),
        (GUEST_STACK_TOP - PAGE_BYTES as u64, stack_pa.as_u64()),
        (GUEST_CODE, code_pa0.as_u64()),
        (GUEST_CODE + PAGE_BYTES as u64, code_pa1.as_u64()),
        (GUEST_PML4, pml4_pa.as_u64()),
        (GUEST_PDPT, pdpt_pa.as_u64()),
        (GUEST_PD, pd_pa.as_u64()),
    ];
    for (gpa, hpa) in mappings {
        if ept::map(
            &mut memory,
            root,
            gpa,
            hpa,
            GUEST_EPT_ACCESS,
            EptMemoryType::WriteBack,
        )
        .is_err()
        {
            error!("vmx: own-address EPT mapping of {gpa:#x} failed");
            return false;
        }
    }
    let eptp = ept::pointer(root);

    let rip = GUEST_CODE | (fn_va & 0xFFF);
    // SAFETY: `cell` is the current VMCS and VMX operation is live; the guest's
    // code, stack and data are reachable through the EPT and its own page tables
    // built above, and the CR3 override points at the guest's PML4.
    let programmed = unsafe {
        program_guest(cell, Some(eptp), rip, GUEST_STACK_TOP)
            .and_then(|()| cell.write(Field::GUEST_CR3, GUEST_PML4))
    };
    if let Err(error) = programmed {
        error!("vmx: own-address programming failed: {error}");
        return false;
    }

    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let ok = reached && registers.r8 == GUEST_PLANTED && registers.r9 == GUEST_WRITTEN;
    if !ok {
        error!(
            "vmx: own-address: reached {reached}, planted {:#x}, written {:#x}",
            registers.r8, registers.r9
        );
    }
    // The EPT tables, guest paging frames and data/stack pages are all walked by
    // the processor for the whole of the run above, so they are dropped only now.
    drop((data, stack, pml4, pdpt, pd, memory));
    ok
}

/// The marker the injected-event handler writes, recognizable and spanning all
/// eight bytes.
const INJECT_MARKER: u64 = 0x1515_2020_2525_3030;

/// The handler the injected events vector to: it records a marker and
/// `VMCALL`s.
///
/// Reaching it is the whole of the proof — the only way control arrives here is
/// the processor delivering the injected event through the gate the host placed
/// in the guest's IDT.
#[unsafe(naked)]
unsafe extern "C" fn inject_handler() {
    core::arch::naked_asm!(
        "mov r8, {marker}",
        "vmcall",
        "2:",
        "hlt",
        "jmp 2b",
        marker = const INJECT_MARKER,
    );
}

/// A guest whose own code never runs, because the event injected on entry is
/// delivered before its first instruction.
#[unsafe(naked)]
unsafe extern "C" fn guest_idle() {
    core::arch::naked_asm!("2:", "hlt", "jmp 2b");
}

/// Writes a 64-bit interrupt gate for `vector` into `idt`, pointing at
/// `handler` with code selector `cs`, present and ring 0.
fn write_gate(idt: &mut Page, vector: u8, handler: u64, cs: u16) {
    let base = usize::from(vector) * 16;
    idt.0[base..base + 2].copy_from_slice(&((handler & 0xFFFF) as u16).to_le_bytes());
    idt.0[base + 2..base + 4].copy_from_slice(&cs.to_le_bytes());
    idt.0[base + 4] = 0;
    // Present, descriptor-privilege zero, 64-bit interrupt-gate type.
    idt.0[base + 5] = 0x8E;
    idt.0[base + 6..base + 8].copy_from_slice(&(((handler >> 16) & 0xFFFF) as u16).to_le_bytes());
    idt.0[base + 8..base + 12]
        .copy_from_slice(&(((handler >> 32) & 0xFFFF_FFFF) as u32).to_le_bytes());
    idt.0[base + 12..base + 16].copy_from_slice(&0u32.to_le_bytes());
}

/// Injects `vector` into a guest that owns an IDT whose gate for it points at
/// [`inject_handler`], and checks the handler ran.
///
/// The guest runs in the host's address space with no second translation, so
/// its IDT, the handler and the stack the processor pushes the interrupt frame
/// onto are all reachable directly. `error_code` injects the event as one that
/// carries an error code, which the processor pushes before the frame.
fn inject_probe(cell: &mut Vmcs, vector: u8, error_code: bool) -> bool {
    let mut idt = Page::zeroed();
    let stack = Page::zeroed();
    let handler = (inject_handler as *const ()).addr() as u64;
    write_gate(&mut idt, vector, handler, CS::get_reg().0);
    let idt_base = core::ptr::from_ref(&idt.0).addr() as u64;

    let rip = (guest_idle as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    let event = Interruption::inject(vector, Kind::HardwareException, error_code);
    // SAFETY: `cell` is the current VMCS and VMX operation is live; the guest
    // shares the host address space with no EPT, so `rip`, the IDT at `idt_base`
    // and `rsp` are all host-mapped, and the injected vector has a present gate.
    let programmed = unsafe {
        program_guest(cell, None, rip, rsp)
            .and_then(|()| cell.write(Field::GUEST_IDTR_BASE, idt_base))
            .and_then(|()| cell.write(Field::GUEST_IDTR_LIMIT, 0xFFF))
            .and_then(|()| cell.write(Field::VM_ENTRY_INTERRUPTION_INFO, u64::from(event.bits())))
            .and_then(|()| {
                if error_code {
                    cell.write(Field::VM_ENTRY_EXCEPTION_ERROR_CODE, 0)
                } else {
                    Ok(())
                }
            })
    };
    if let Err(error) = programmed {
        error!("vmx: injection programming failed: {error}");
        drop((idt, stack));
        return false;
    }

    let mut registers = Registers::default();
    let reached = drive_dispatch(cell, &mut registers);
    let ok = reached && registers.r8 == INJECT_MARKER;
    if !ok {
        error!(
            "vmx: injection vector {vector}: reached {reached}, marker {:#x}",
            registers.r8
        );
    }
    drop((idt, stack));
    ok
}

/// The basic exit reason the current VMCS records, or `None` if it cannot be
/// read.
fn exit_reason(cell: &Vmcs) -> Option<BasicExitReason> {
    // SAFETY: `cell` is the current VMCS on this processor.
    let bits = unsafe { cell.read(Field::EXIT_REASON) }.ok()?;
    Some(ExitReason::from_bits((bits & 0xFFFF_FFFF) as u32).basic())
}

/// Bit 13 of `CR4`: the virtual-machine-extensions enable.
const CR4_VMXE: u64 = 1 << 13;

/// Whether `CR4.VMXE` is set, which the enable path must have done for `VMXON`
/// to have succeeded.
fn cr4_vmxe_set() -> bool {
    x86_64::registers::control::Cr4::read_raw() & CR4_VMXE != 0
}

/// Whether writing `value` to `field` and reading it back returns `value`.
fn roundtrip(cell: &Vmcs, field: vmx::FieldEncoding, value: u64) -> bool {
    // SAFETY: `cell` is the current VMCS on this processor.
    let result = unsafe { cell.write(field, value).and_then(|()| cell.read(field)) };
    matches!(result, Ok(read) if read == value)
}

/// Leaves VMX operation, logging if the instruction is refused.
///
/// # Safety
///
/// This processor must be in VMX operation with no current VMCS.
unsafe fn leave() {
    // SAFETY: the caller guarantees VMX operation with no current VMCS.
    if let Err(error) = unsafe { instr::vmxoff() }.ok() {
        error!("vmx: VMXOFF failed: {error}");
    }
}
