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
use vmx::{
    BasicExitReason, EptEntry, EptPointer, ExitReason, Field, Interruption, PAGE_BYTES, VmxBasic,
    event::Kind as EventKind,
};
use x86_64::VirtAddr;

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

    // A battery of checks: each logs PASS or FAIL and the tally at the end sums
    // them, so one boot exercises many things rather than one.
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
        (1..=PAGE_BYTES).contains(&(vmx.basic().region_bytes() as usize)),
    );

    // Field round-trips across every width, which exercises the encoding of
    // each: a wrong width or index would read back something other than what
    // was written.
    check(
        "16-bit field round-trip",
        roundtrip(&cell, Field::GUEST_ES_SELECTOR, 0x1234),
    );
    check(
        "32-bit field round-trip",
        roundtrip(&cell, Field::GUEST_ES_LIMIT, 0xDEAD_BEEF),
    );
    check(
        "64-bit field round-trip",
        roundtrip(&cell, Field::TSC_OFFSET, 0x1122_3344_5566_7788),
    );
    check(
        "natural-width field round-trip",
        roundtrip(&cell, Field::GUEST_RIP, 0x0000_0000_0040_1000),
    );
    check(
        "guest launch, resume across CPUIDs, VMCALL",
        entry_probe(&mut cell),
    );
    check(
        "guest GPRs saved on exit (CPUID results)",
        register_save_probe(&mut cell),
    );
    check(
        "event injection (#UD delivered and intercepted)",
        injection_probe(&mut cell),
    );
    // SAFETY: we reached here only by entering VMX operation, so this is a
    // VMX-capable processor and the capability registers exist.
    if unsafe { controls::ept_available() } {
        check(
            "guest launch under EPT (second translation)",
            ept_probe(&mut cell, space),
        );
    } else {
        warn!("vmx: EPT not available on this processor; skipping the EPT guest check");
    }
    check(
        "second VMCS switch and independence",
        second_vmcs_probe(&cell, space, vmx.basic()),
    );

    info!("vmx: SELF-TEST SUMMARY: {passed}/{total} checks passed");

    // SAFETY: `cell` is current, so clearing it leaves no current VMCS, which is
    // the precondition for leaving VMX operation.
    unsafe {
        let _ = instr::vmclear(cell.region()).ok();
        leave();
    }
    info!("vmx: self-test complete");
    true
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

/// Launches the guest once with its registers cleared and checks the world
/// switch brought the `CPUID` results back.
///
/// With `RAX` zero the guest runs `CPUID` leaf 0, which returns the highest
/// leaf in `RAX` and the vendor string in `RBX`/`RDX`/`RCX`. Seeing those
/// non-zero after the exit means the run loop saved the guest's general
/// registers, not just entered and left.
fn register_save_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
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
        && exit_reason(cell) == Some(BasicExitReason::CPUID)
        && registers.rax != 0
        && registers.rbx != 0;
    if !ok {
        error!(
            "vmx: register-save: {entered:?}, rax {:#x}, rbx {:#x}",
            registers.rax, registers.rbx
        );
    }
    drop(stack);
    ok
}

/// Bit of the exception bitmap and the vector of the invalid-opcode exception,
/// which carries no error code and needs no instruction length to inject.
const INVALID_OPCODE_VECTOR: u8 = 6;

/// Injects a `#UD` into the guest and checks it was delivered and intercepted.
///
/// The exception bitmap is set to intercept the vector, and the entry
/// interruption-information field is set to inject it; a correct injection
/// delivers the exception on entry, which the bitmap turns into an exit whose
/// interruption-information field names the same vector.
fn injection_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    let event = Interruption::inject(INVALID_OPCODE_VECTOR, EventKind::HardwareException, false);
    // SAFETY: `cell` is current and VMX operation is live.
    if let Err(error) = unsafe {
        program_guest(cell, None, rip, rsp)
            .and_then(|()| cell.write(Field::EXCEPTION_BITMAP, 1_u64 << INVALID_OPCODE_VECTOR))
            .and_then(|()| cell.write(Field::VM_ENTRY_INTERRUPTION_INFO, u64::from(event.bits())))
    } {
        error!("vmx: injection programming failed: {error}");
        drop(stack);
        return false;
    }

    // SAFETY: `cell` is current and programmed.
    let entered = unsafe { run::run(cell, &mut Registers::default()) };
    let ok = match entered {
        Entered::Exited if exit_reason(cell) == Some(BasicExitReason::EXCEPTION_OR_NMI) => {
            // SAFETY: `cell` is current.
            let info = unsafe { cell.read(Field::VM_EXIT_INTERRUPTION_INFO) }.unwrap_or(0);
            let delivered = Interruption::from_bits((info & 0xFFFF_FFFF) as u32);
            delivered.is_valid() && delivered.vector() == INVALID_OPCODE_VECTOR
        }
        Entered::Exited => {
            error!(
                "vmx: injected #UD but exited with reason {:?}",
                exit_reason(cell)
            );
            false
        }
        Entered::Failed(fail) => {
            error!("vmx: injection entry rejected ({fail})");
            false
        }
    };
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
