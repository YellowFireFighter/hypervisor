//! Refusing the VMX instructions a concealed guest must not find.
//!
//! The guest is told through `CPUID` that there is no virtualization extension,
//! so every VMX instruction it executes must look as though the processor has
//! none. In VMX non-root operation those instructions do not fault — they exit
//! to the host unconditionally — so the fault the guest should have taken is
//! delivered here instead: an invalid-opcode exception through the guest's own
//! descriptor table, with its instruction pointer left on the refused
//! instruction so the fault appears to come from it.
//!
//! This is the VMX counterpart to the AMD side's refusal of SVM instructions.

use vmcs::{Vmcs, error::VmFail};
use vmx::{Field, Interruption, event::Kind};

/// The invalid-opcode exception vector.
const INVALID_OPCODE: u8 = 6;

/// Delivers `#UD` to the guest for an instruction it has been told cannot
/// exist, leaving its instruction pointer on that instruction.
///
/// # Errors
///
/// The [`VmFail`] from the `VMWRITE` of the entry interruption field.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, in VMX operation.
pub(crate) unsafe fn refuse(cell: &Vmcs) -> Result<(), VmFail> {
    let event = Interruption::inject(INVALID_OPCODE, Kind::HardwareException, false);
    // SAFETY: the caller guarantees the current VMCS. A hardware exception
    // needs no error code and no instruction length, and the instruction
    // pointer is left unchanged, so the fault is taken on the refused
    // instruction and delivered through the guest's own IDT on the next entry.
    unsafe { cell.write(Field::VM_ENTRY_INTERRUPTION_INFO, u64::from(event.bits())) }
}
