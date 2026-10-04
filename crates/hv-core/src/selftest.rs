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

use log::{error, info};
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

    // A natural-width guest-state field that stores and reads back any canonical
    // value; the probe is page-aligned and canonical.
    let field = Field::GUEST_RIP;
    let probe: u64 = 0x0000_0000_0040_1000;
    // SAFETY: `cell` is the current VMCS on this processor.
    let outcome = unsafe { cell.write(field, probe).and_then(|()| cell.read(field)) };
    match outcome {
        Ok(read) if read == probe => {
            info!("vmx: SELF-TEST PASS: VMWRITE/VMREAD round-tripped {probe:#x}");
        }
        Ok(read) => error!("vmx: SELF-TEST FAIL: round-trip read {read:#x}, expected {probe:#x}"),
        Err(error) => error!("vmx: SELF-TEST FAIL: VMWRITE/VMREAD: {error}"),
    }

    entry_probe(&mut cell);

    // SAFETY: `cell` is current, so clearing it leaves no current VMCS, which is
    // the precondition for leaving VMX operation.
    unsafe {
        let _ = instr::vmclear(cell.region()).ok();
        leave();
    }
    info!("vmx: self-test complete");
    true
}

/// A guest whose first instruction exits unconditionally.
///
/// `CPUID` always causes a VM exit from non-root operation, so a guest entered
/// here runs exactly one instruction before coming back — which is the whole of
/// what the entry probe needs. It is a naked function so its address is in the
/// image's executable text, which the guest reaches through the host's own page
/// tables; the loop after `CPUID` is never run, because the exit does not
/// resume.
#[unsafe(naked)]
unsafe extern "C" fn guest_probe() {
    core::arch::naked_asm!("cpuid", "2:", "jmp 2b");
}

/// Programs a flat 64-bit guest, enters it, and reports how the entry ended.
///
/// This is the first exercise of the world switch and the VM-entry checks: a
/// clean `GUEST-ENTRY PASS` means a guest was launched, ran its `CPUID`, and
/// exited back into the hypervisor. A failure prints the VM-instruction-error
/// or the unexpected exit reason, which is what narrows down a wrong field.
fn entry_probe(cell: &mut Vmcs) {
    let stack = Page::zeroed();
    let rip = (guest_probe as *const ()).addr() as u64;
    let rsp = core::ptr::from_ref(&stack.0).addr() as u64 + PAGE_BYTES as u64;

    // SAFETY: `cell` is the current VMCS on this processor and VMX operation is
    // live. The programming captures this running host, sets a flat 64-bit
    // guest in the host's own address space with `rip` in executable image text
    // and `rsp` in the freshly allocated stack page, and the run loop preserves
    // and restores the host around the entry.
    let entered = unsafe {
        match host::program(cell)
            .and_then(|()| controls::program(cell))
            .and_then(|()| guest::program(cell, rip, rsp))
        {
            Ok(()) => Some(run::run(cell, &mut Registers::default())),
            Err(error) => {
                error!("vmx: GUEST-ENTRY FAIL: VMCS programming: {error}");
                None
            }
        }
    };

    match entered {
        Some(Entered::Exited) => report_exit(cell),
        Some(Entered::Failed(fail)) => {
            // SAFETY: `cell` is still current.
            let number = unsafe { cell.read(Field::VM_INSTRUCTION_ERROR) }.unwrap_or(0);
            error!("vmx: GUEST-ENTRY FAIL: entry rejected ({fail}); VM-instruction-error {number}");
        }
        None => {}
    }
    // The stack page is live for the whole entry above; keeping it named here
    // makes that lifetime explicit.
    drop(stack);
}

/// Reports the reason a guest exited, as a `PASS` when it was the expected
/// `CPUID`.
fn report_exit(cell: &Vmcs) {
    // SAFETY: `cell` is the current VMCS on this processor.
    let reason = unsafe { cell.read(Field::EXIT_REASON) };
    match reason {
        Ok(bits) => {
            let exit = ExitReason::from_bits((bits & 0xFFFF_FFFF) as u32);
            if exit.basic() == BasicExitReason::CPUID {
                info!(
                    "vmx: GUEST-ENTRY PASS: guest launched, ran CPUID and exited (reason {})",
                    exit.basic().number()
                );
            } else {
                info!(
                    "vmx: guest exited with reason {} (expected CPUID = 10)",
                    exit.basic().number()
                );
            }
        }
        Err(error) => error!("vmx: guest exited but EXIT_REASON could not be read: {error}"),
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
        error!("vmx: VMXOFF failed: {error}");
    }
}
