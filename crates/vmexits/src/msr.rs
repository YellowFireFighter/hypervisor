//! Answering a guest's `RDMSR` and `WRMSR`, with virtualization concealed.
//!
//! Most registers are forwarded to the machine's own. The index the guest named
//! is its `ECX`; a read returns the register in `EDX:EAX`, and a write takes
//! the value from `EDX:EAX` — the halves the architecture splits a
//! sixty-four-bit register into, which is why each handler assembles or splits
//! the two thirty-two-bit registers rather than using the whole of `RAX` and
//! `RDX`.
//!
//! The forwarding goes through [`probe`] rather than a bare instruction because
//! most of the indices a guest can name are not registers, and the only way to
//! find out is to attempt the access and catch the fault. A refusal is reported
//! back as the index that was refused; the general-protection fault the guest
//! is owed for it needs the event injection a later layer builds.
//!
//! One register is not forwarded. `IA32_FEATURE_CONTROL` is where firmware
//! records whether virtualization may be used, and it is the register firmware
//! looks at after `CPUID` has denied the extension. It is answered as a machine
//! whose firmware turned virtualization off and locked that decision — the lock
//! set and both `VMXON` permissions clear — which is the same story the cleared
//! `CPUID` bit tells, and a state guests already know how to be told about. A
//! write to it is swallowed rather than forwarded: the guest is shown a locked
//! register, and a locked register ignores writes rather than taking the fault
//! the real one would raise.

use vmcs::Registers;

/// `IA32_FEATURE_CONTROL`, the register firmware locks virtualization behind.
const IA32_FEATURE_CONTROL: u32 = 0x3A;
/// The value `IA32_FEATURE_CONTROL` is answered with: bit 0, the lock, set, and
/// the two `VMXON` permission bits (1 and 2) clear — virtualization present in
/// silicon but turned off by firmware and the decision locked.
const FEATURE_CONTROL_CONCEALED: u64 = 1 << 0;

/// Reads the model-specific register the guest named into its `EDX:EAX`.
///
/// # Errors
///
/// The index, if the machine has no such register.
pub(crate) fn read(registers: &mut Registers) -> Result<(), u32> {
    let msr = (registers.rcx & 0xFFFF_FFFF) as u32;
    let value = if msr == IA32_FEATURE_CONTROL {
        FEATURE_CONTROL_CONCEALED
    } else {
        probe::read(msr).map_err(|_| msr)?
    };
    registers.rax = value & 0xFFFF_FFFF;
    registers.rdx = value >> 32;
    Ok(())
}

/// Writes the guest's `EDX:EAX` into the model-specific register it named.
///
/// A write to the concealed `IA32_FEATURE_CONTROL` is swallowed, because the
/// guest is shown a locked register.
///
/// # Errors
///
/// The index, if the machine has no such register.
pub(crate) fn write(registers: &Registers) -> Result<(), u32> {
    let msr = (registers.rcx & 0xFFFF_FFFF) as u32;
    if msr == IA32_FEATURE_CONTROL {
        return Ok(());
    }
    let value = ((registers.rdx & 0xFFFF_FFFF) << 32) | (registers.rax & 0xFFFF_FFFF);
    probe::write(msr, value).map_err(|_| msr)
}
