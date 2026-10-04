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
    Capability, EptPointer, Field, PrimaryProc, SecondaryProc, VmEntry, VmExit,
    control::{
        IA32_VMX_ENTRY_CTLS, IA32_VMX_EXIT_CTLS, IA32_VMX_PINBASED_CTLS, IA32_VMX_PROCBASED_CTLS,
        IA32_VMX_PROCBASED_CTLS2,
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
