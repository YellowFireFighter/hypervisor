//! The execution, exit and entry controls a VM entry is governed by.
//!
//! Each of the control words is reconciled against its capability register —
//! the forced bits set, the forbidden bits cleared — before it is written,
//! because a control bit the processor does not allow fails the entry. On top
//! of the forced bits this asks for only what a 64-bit host running a 64-bit
//! guest needs: the host stays in 64-bit mode across the exit, the guest enters
//! in 64-bit mode, and `IA32_EFER` is loaded on both sides.
//!
//! It also writes the handful of adjacent fields a valid VMCS requires to be
//! set even when nothing uses them: the counts of the register lists (none),
//! the exception bitmap (empty), the control-register masks (the guest owns its
//! own bits) and the VMCS link pointer (no shadow).

use vmx::{
    Capability, EptPointer, Field, PinBased, PrimaryProc, SecondaryProc, VmEntry, VmExit, VmxBasic,
    basic::IA32_VMX_BASIC,
    control::{
        IA32_VMX_ENTRY_CTLS, IA32_VMX_EXIT_CTLS, IA32_VMX_PINBASED_CTLS, IA32_VMX_PROCBASED_CTLS,
        IA32_VMX_PROCBASED_CTLS2, IA32_VMX_TRUE_PROCBASED_CTLS,
    },
};

use crate::{Vmcs, error::VmFail, msr};

/// A VMCS link pointer naming no shadow VMCS, which is what an ordinary VMCS
/// carries.
const NO_SHADOW_VMCS: u64 = !0;

/// Programs the control fields of the current VMCS for a 64-bit guest.
///
/// With `ept` set, the guest runs behind that second translation: the secondary
/// controls are activated, EPT is enabled, and the pointer is installed. With
/// it `None`, the guest runs in the host's own address space and no secondary
/// control is asked for unless the processor forces one.
///
/// # Errors
///
/// The [`VmFail`] from the first `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, which must be in VMX
/// operation.
pub unsafe fn program(cell: &Vmcs, ept: Option<EptPointer>) -> Result<(), VmFail> {
    // SAFETY: the caller guarantees the current VMCS and VMX operation; the
    // capability registers read here exist on a VMX-capable processor.
    unsafe {
        let pin = Capability::from_bits(msr::rdmsr(IA32_VMX_PINBASED_CTLS));
        let primary_cap = Capability::from_bits(msr::rdmsr(IA32_VMX_PROCBASED_CTLS));
        let exit = Capability::from_bits(msr::rdmsr(IA32_VMX_EXIT_CTLS));
        let entry = Capability::from_bits(msr::rdmsr(IA32_VMX_ENTRY_CTLS));

        cell.reconcile_control(Field::PIN_BASED_CONTROLS, 0, pin)?;

        // EPT lives in the secondary controls, so enabling it means activating
        // them in the primary word.
        let primary_desired = if ept.is_some() {
            PrimaryProc::ACTIVATE_SECONDARY_CONTROLS.bits()
        } else {
            0
        };
        let primary = primary_cap.reconcile(primary_desired);
        cell.write(Field::PRIMARY_PROC_CONTROLS, u64::from(primary))?;
        // The secondary controls are live only when the primary word activates
        // them. Write them only then, and only when the processor has the
        // register that reports them, so a processor without secondary controls
        // is not asked for a reserved model-specific register.
        if primary & PrimaryProc::ACTIVATE_SECONDARY_CONTROLS.bits() != 0 {
            let secondary_cap = Capability::from_bits(msr::rdmsr(IA32_VMX_PROCBASED_CTLS2));
            let secondary_desired = if ept.is_some() {
                SecondaryProc::ENABLE_EPT.bits()
            } else {
                0
            };
            cell.reconcile_control(
                Field::SECONDARY_PROC_CONTROLS,
                secondary_desired,
                secondary_cap,
            )?;
        }
        if let Some(pointer) = ept {
            cell.write(Field::EPT_POINTER, pointer.bits())?;
        }

        cell.reconcile_control(
            Field::PRIMARY_VM_EXIT_CONTROLS,
            VmExit::HOST_ADDRESS_SPACE_SIZE.bits() | VmExit::LOAD_IA32_EFER.bits(),
            exit,
        )?;
        cell.reconcile_control(
            Field::VM_ENTRY_CONTROLS,
            VmEntry::IA32E_MODE_GUEST.bits() | VmEntry::LOAD_IA32_EFER.bits(),
            entry,
        )?;

        cell.write(Field::EXCEPTION_BITMAP, 0)?;
        cell.write(Field::PAGE_FAULT_ERROR_CODE_MASK, 0)?;
        cell.write(Field::PAGE_FAULT_ERROR_CODE_MATCH, 0)?;
        cell.write(Field::CR3_TARGET_COUNT, 0)?;
        cell.write(Field::VM_EXIT_MSR_STORE_COUNT, 0)?;
        cell.write(Field::VM_EXIT_MSR_LOAD_COUNT, 0)?;
        cell.write(Field::VM_ENTRY_MSR_LOAD_COUNT, 0)?;
        cell.write(Field::VM_ENTRY_INTERRUPTION_INFO, 0)?;
        cell.write(Field::TSC_OFFSET, 0)?;
        // The guest owns every bit of its control registers: an empty mask means
        // no guest write to CR0 or CR4 exits, and the read shadows are then
        // never consulted.
        cell.write(Field::CR0_GUEST_HOST_MASK, 0)?;
        cell.write(Field::CR4_GUEST_HOST_MASK, 0)?;
        cell.write(Field::VMCS_LINK_POINTER, NO_SHADOW_VMCS)?;
    }
    Ok(())
}

/// Whether this processor can run a guest behind EPT: the primary controls must
/// allow the secondary word, and the secondary word must allow EPT.
///
/// # Safety
///
/// This processor must support VMX, so the capability registers read here
/// exist.
#[must_use]
pub unsafe fn ept_available() -> bool {
    // SAFETY: the caller guarantees a VMX-capable processor, on which these
    // capability registers exist.
    unsafe {
        let primary = Capability::from_bits(msr::rdmsr(IA32_VMX_PROCBASED_CTLS));
        if !primary.allows(PrimaryProc::ACTIVATE_SECONDARY_CONTROLS.bits()) {
            return false;
        }
        let secondary = Capability::from_bits(msr::rdmsr(IA32_VMX_PROCBASED_CTLS2));
        secondary.allows(SecondaryProc::ENABLE_EPT.bits())
    }
}

/// Stops the guest exiting on its own `CR3` loads and stores, when the
/// processor allows it.
///
/// The non-"true" primary capability register forces `CR3`-load and
/// `CR3`-store exiting on, a legacy of processors without EPT that needed to
/// watch a guest's paging. With EPT the guest owns its `CR3`, and firmware run
/// as a guest reads and writes `CR3` constantly — every exception entry saves
/// it — so exiting on each is both needless and fatal to a guest that has no
/// handler for it yet. A processor that reports the "true" capability registers
/// (`IA32_VMX_BASIC` bit 55) is allowed to clear these bits; one that does not
/// cannot, and this leaves them set and answers `false`.
///
/// Call it after [`program`], whose primary-control word this rewrites.
///
/// # Errors
///
/// The [`VmFail`] from the `VMREAD` or `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn relax_cr3_exiting(cell: &Vmcs) -> Result<bool, VmFail> {
    // SAFETY: the caller guarantees VMX operation, on which these capability
    // registers exist.
    if !unsafe { VmxBasic::from_bits(msr::rdmsr(IA32_VMX_BASIC)) }.has_true_controls() {
        return Ok(false);
    }
    // SAFETY: the true capability register exists once bit 55 is set.
    let capability = Capability::from_bits(unsafe { msr::rdmsr(IA32_VMX_TRUE_PROCBASED_CTLS) });
    let cr3_exiting = PrimaryProc::CR3_LOAD_EXITING.bits() | PrimaryProc::CR3_STORE_EXITING.bits();
    if capability.forces(cr3_exiting) {
        return Ok(false);
    }
    // SAFETY: the caller guarantees the current VMCS; the word is the one
    // `program` already reconciled and wrote, with only the two CR3 bits
    // cleared, which the true capability register above permits.
    unsafe {
        let current = u32::try_from(cell.read(Field::PRIMARY_PROC_CONTROLS)?).unwrap_or(0);
        cell.write(
            Field::PRIMARY_PROC_CONTROLS,
            u64::from(current & !cr3_exiting),
        )?;
    }
    Ok(true)
}

/// Turns on APIC virtualization for the current VMCS, pointing it at a
/// virtual-APIC page the guest's register accesses are served from and an
/// APIC-access page whose guest-physical mapping the processor watches.
///
/// With this on, the guest's reads of its local APIC, and most of its writes,
/// are satisfied against the virtual-APIC page without reaching the real
/// controller; the writes that still need the host — the interrupt command
/// among them — leave the value in the page and exit with
/// [`APIC_WRITE`](vmx::BasicExitReason::APIC_WRITE). The processor must allow
/// both the access and the register-virtualization controls — a VMX-capable
/// machine with an on-die APIC does — and this leaves whichever it is refused
/// clear, so the caller checks the result it reads back if it must.
///
/// Call it after [`program`], whose primary and secondary control words it adds
/// to. The EPT must map the guest's APIC page to `access_phys` as a 4-KiB leaf,
/// which is what makes an access to it recognizable.
///
/// # Errors
///
/// The [`VmFail`] from the first field access the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn virtualize_apic(
    cell: &Vmcs,
    vapic_phys: u64,
    access_phys: u64,
) -> Result<(), VmFail> {
    // SAFETY: the caller guarantees the current VMCS in VMX operation, on which
    // the capability register read here exists.
    unsafe {
        let primary = cell.read(Field::PRIMARY_PROC_CONTROLS)?;
        cell.write(
            Field::PRIMARY_PROC_CONTROLS,
            primary
                | u64::from(
                    PrimaryProc::USE_TPR_SHADOW.bits()
                        | PrimaryProc::ACTIVATE_SECONDARY_CONTROLS.bits(),
                ),
        )?;
        let capability = Capability::from_bits(msr::rdmsr(IA32_VMX_PROCBASED_CTLS2));
        let existing = u32::try_from(cell.read(Field::SECONDARY_PROC_CONTROLS)?).unwrap_or(0);
        let desired = existing
            | SecondaryProc::VIRTUALIZE_APIC_ACCESSES.bits()
            | SecondaryProc::APIC_REGISTER_VIRTUALIZATION.bits();
        cell.write(
            Field::SECONDARY_PROC_CONTROLS,
            u64::from(capability.reconcile(desired)),
        )?;
        cell.write(Field::VIRTUAL_APIC_ADDR, vapic_phys)?;
        cell.write(Field::APIC_ACCESS_ADDR, access_phys)?;
        cell.write(Field::TPR_THRESHOLD, 0)?;
    }
    Ok(())
}

/// Whether this processor can virtualize a guest's APIC accesses and registers,
/// which [`virtualize_apic`] needs.
///
/// # Safety
///
/// This processor must support VMX, so the capability register read here
/// exists.
#[must_use]
pub unsafe fn apic_virtualization_available() -> bool {
    // SAFETY: the caller guarantees a VMX-capable processor.
    let capability = Capability::from_bits(unsafe { msr::rdmsr(IA32_VMX_PROCBASED_CTLS2) });
    capability.allows(
        SecondaryProc::VIRTUALIZE_APIC_ACCESSES.bits()
            | SecondaryProc::APIC_REGISTER_VIRTUALIZATION.bits(),
    )
}

/// Arms the VMX-preemption timer, so the guest exits after it counts `value`
/// down to zero and the exit can be acted on — sampling where a guest that
/// otherwise never exits is spending its time, above all.
///
/// The value reloads from the field on every entry, so a caller that re-enters
/// without rewriting it is preempted again each quantum. The pin control is
/// reconciled against the capability register, so a processor that does not
/// offer the timer is left without it rather than refused entry.
///
/// # Errors
///
/// The [`VmFail`] from the first field access the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn set_preemption_timer(cell: &Vmcs, value: u32) -> Result<(), VmFail> {
    // SAFETY: the caller guarantees the current VMCS in VMX operation, on which
    // the capability register exists.
    unsafe {
        let capability = Capability::from_bits(msr::rdmsr(IA32_VMX_PINBASED_CTLS));
        let existing = u32::try_from(cell.read(Field::PIN_BASED_CONTROLS)?).unwrap_or(0);
        let reconciled =
            capability.reconcile(existing | PinBased::ACTIVATE_PREEMPTION_TIMER.bits());
        cell.write(Field::PIN_BASED_CONTROLS, u64::from(reconciled))?;
        cell.write(Field::VMX_PREEMPTION_TIMER_VALUE, u64::from(value))?;
    }
    Ok(())
}

/// Makes external interrupts exit to the host, with the interrupt acknowledged
/// on exit so the exit carries its vector, and reports whether the processor
/// allows both.
///
/// This is how the host takes a guest's interrupts for itself: every physical
/// interrupt becomes a VM exit whose interruption-information field already
/// names the vector the real controller delivered, which the host then reflects
/// into the guest through the guest's own interrupt descriptor table — the VMX
/// counterpart of the AMD world switch's interrupt interception. A processor
/// that forbids either control is left without the feature rather than refused
/// entry, and answers `false`.
///
/// Call it after [`program`], whose pin and exit control words this adds to.
///
/// # Errors
///
/// The [`VmFail`] from the `VMREAD` or `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn reflect_interrupts(cell: &Vmcs) -> Result<bool, VmFail> {
    // SAFETY: the caller guarantees the current VMCS in VMX operation, on which
    // these capability registers exist.
    unsafe {
        let pin_cap = Capability::from_bits(msr::rdmsr(IA32_VMX_PINBASED_CTLS));
        let exit_cap = Capability::from_bits(msr::rdmsr(IA32_VMX_EXIT_CTLS));
        if !pin_cap.allows(PinBased::EXTERNAL_INTERRUPT_EXITING.bits())
            || !exit_cap.allows(VmExit::ACKNOWLEDGE_INTERRUPT_ON_EXIT.bits())
        {
            return Ok(false);
        }
        let pin = cell.read(Field::PIN_BASED_CONTROLS)?;
        cell.write(
            Field::PIN_BASED_CONTROLS,
            pin | u64::from(PinBased::EXTERNAL_INTERRUPT_EXITING.bits()),
        )?;
        let exit = cell.read(Field::PRIMARY_VM_EXIT_CONTROLS)?;
        cell.write(
            Field::PRIMARY_VM_EXIT_CONTROLS,
            exit | u64::from(VmExit::ACKNOWLEDGE_INTERRUPT_ON_EXIT.bits()),
        )?;
    }
    Ok(true)
}

/// Turns interrupt-window exiting on or off.
///
/// With it on, the guest exits the moment it can take an interrupt — interrupts
/// enabled and no instruction shadow — which is how the host delivers an
/// interrupt it is holding for a guest that had interrupts masked when the
/// interrupt arrived. It is turned off again once nothing is waiting, so a
/// guest that can always take its interrupts does not exit for a window it does
/// not need.
///
/// # Errors
///
/// The [`VmFail`] from the `VMREAD` or `VMWRITE` the processor rejects.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub unsafe fn request_interrupt_window(cell: &Vmcs, want: bool) -> Result<(), VmFail> {
    let bit = u64::from(PrimaryProc::INTERRUPT_WINDOW_EXITING.bits());
    // SAFETY: the caller guarantees the current VMCS in VMX operation.
    unsafe {
        let primary = cell.read(Field::PRIMARY_PROC_CONTROLS)?;
        let next = if want { primary | bit } else { primary & !bit };
        cell.write(Field::PRIMARY_PROC_CONTROLS, next)?;
    }
    Ok(())
}
