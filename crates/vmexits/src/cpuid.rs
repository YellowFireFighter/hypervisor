//! Answering a guest's `CPUID`.
//!
//! `CPUID` always exits, and the guest's registers do not reflect its result
//! because the exit is taken before the instruction executes: the host runs it
//! on the guest's behalf and places the four words it produced into the guest's
//! registers, which the world switch loads back on the next entry.
//!
//! This forwards the processor's own answer. The concealment the AMD side
//! applies — clearing the hypervisor-present bit and emptying the hypervisor
//! leaves — is a policy edit on top of this, and belongs with the rest of the
//! stealth decisions rather than in the mechanics of taking the exit.

use vmcs::Registers;

/// Runs the guest's `CPUID` and writes its result back into the guest's
/// registers.
///
/// The leaf is the guest's `EAX` and the subleaf its `ECX`, both of which the
/// world switch saved on the exit. Each result word is written as the
/// thirty-two bits `CPUID` produces, zero-extended into the sixty-four-bit
/// register exactly as the instruction would have left it.
pub(crate) fn emulate(registers: &mut Registers) {
    let leaf = (registers.rax & 0xFFFF_FFFF) as u32;
    let subleaf = (registers.rcx & 0xFFFF_FFFF) as u32;
    let result = processor::cpuid(leaf, subleaf);
    registers.rax = u64::from(result.eax);
    registers.rbx = u64::from(result.ebx);
    registers.rcx = u64::from(result.ecx);
    registers.rdx = u64::from(result.edx);
}
