//! Answering a guest's `RDMSR` and `WRMSR`.
//!
//! Both are forwarded to the machine's own register. The index the guest named
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

use vmcs::Registers;

/// Reads the model-specific register the guest named into its `EDX:EAX`.
///
/// # Errors
///
/// The index, if the machine has no such register.
pub(crate) fn read(registers: &mut Registers) -> Result<(), u32> {
    let msr = (registers.rcx & 0xFFFF_FFFF) as u32;
    let value = probe::read(msr).map_err(|_| msr)?;
    registers.rax = value & 0xFFFF_FFFF;
    registers.rdx = value >> 32;
    Ok(())
}

/// Writes the guest's `EDX:EAX` into the model-specific register it named.
///
/// # Errors
///
/// The index, if the machine has no such register.
pub(crate) fn write(registers: &Registers) -> Result<(), u32> {
    let msr = (registers.rcx & 0xFFFF_FFFF) as u32;
    let value = ((registers.rdx & 0xFFFF_FFFF) << 32) | (registers.rax & 0xFFFF_FFFF);
    probe::write(msr, value).map_err(|_| msr)
}
