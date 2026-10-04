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

use alloc::boxed::Box;

use log::{error, info, warn};
use paging::AddressSpace;
use vmcs::{Entered, Registers, Vmcs, controls, guest, host, instr, run};
use vmx::{BasicExitReason, ExitReason, Field, PAGE_BYTES};
use x86_64::VirtAddr;

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

/// Programs a flat 64-bit guest, runs it through its exits, and reports how it
/// went.
///
/// A clean `GUEST-ENTRY PASS` means a guest was launched, resumed across its
/// `CPUID` exits, and reached its `VMCALL` — so the world switch, both entry
/// instructions, and the RIP advance all work. A failure prints the
/// VM-instruction-error or the unexpected exit reason, which narrows down the
/// wrong field.
fn entry_probe(cell: &mut Vmcs) -> bool {
    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS on this processor and VMX operation is
    // live. The programming captures this running host, sets a flat 64-bit
    // guest in the host's own address space with `rip` in executable image text
    // and `rsp` in the freshly allocated stack page.
    if let Err(error) = unsafe {
        host::program(cell)
            .and_then(|()| controls::program(cell))
            .and_then(|()| guest::program(cell, rip, rsp))
    } {
        error!("vmx: guest-entry VMCS programming failed: {error}");
        return false;
    }

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
