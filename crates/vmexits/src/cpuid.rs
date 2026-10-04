//! Answering a guest's `CPUID`, with the virtualization extension concealed.
//!
//! `CPUID` always exits, and the guest's registers do not reflect its result
//! because the exit is taken before the instruction executes: the host runs it
//! on the guest's behalf and places the four words it produced into the guest's
//! registers, which the world switch loads back on the next entry.
//!
//! The processor's own answer is forwarded, less what would reveal the
//! hypervisor. Every use of the virtualization extension is intercepted, so a
//! guest told the extension exists would be told a thing it cannot act on and
//! would fail somewhere further away than here; the bit that announces it is
//! cleared, as is the hypervisor-present bit beside it and the whole range of
//! leaves reserved for a hypervisor to answer in. This is the VMX counterpart
//! to the AMD side's concealment; the fuller policy there — reconciling the
//! `OSXSAVE`, `APIC` and x2APIC bits against the guest's own control-register
//! and controller state — belongs with that state, which this layer does not
//! yet carry.

use vmcs::Registers;

/// The standard feature leaf, whose `ECX` word carries the virtualization and
/// hypervisor-present bits.
const STANDARD_FEATURES: u32 = 0x1;
/// `CPUID.01H:ECX[5]`: the VMX virtualization extension.
const VMX: u32 = 1 << 5;
/// `CPUID.01H:ECX[31]`: the hypervisor-present bit, which software checks
/// before it goes looking for the hypervisor leaves.
const HYPERVISOR_PRESENT: u32 = 1 << 31;

/// The first leaf of the range reserved for hypervisor use. Neither vendor
/// assigns architectural meaning here, so a guest that probes it should find it
/// as empty as the cleared hypervisor-present bit says it is.
const HYPERVISOR_LEAF_BASE: u32 = 0x4000_0000;
/// The last leaf of that range a guest might plausibly probe.
const HYPERVISOR_LEAF_LIMIT: u32 = 0x4000_00FF;

/// Runs the guest's `CPUID`, conceals the virtualization extension in the
/// result, and writes it back into the guest's registers.
///
/// The leaf is the guest's `EAX` and the subleaf its `ECX`, both of which the
/// world switch saved on the exit. Each result word is written as the
/// thirty-two bits `CPUID` produces, zero-extended into the sixty-four-bit
/// register exactly as the instruction would have left it.
pub(crate) fn emulate(registers: &mut Registers) {
    let leaf = (registers.rax & 0xFFFF_FFFF) as u32;
    let subleaf = (registers.rcx & 0xFFFF_FFFF) as u32;

    // The hypervisor leaves answer as empty rather than with whatever the bare
    // hardware returns there, so a guest that skips the present-bit check and
    // probes the range directly finds nothing to cooperate with.
    if (HYPERVISOR_LEAF_BASE..=HYPERVISOR_LEAF_LIMIT).contains(&leaf) {
        registers.rax = 0;
        registers.rbx = 0;
        registers.rcx = 0;
        registers.rdx = 0;
        return;
    }

    let mut result = processor::cpuid(leaf, subleaf);
    if leaf == STANDARD_FEATURES {
        result.ecx &= !(VMX | HYPERVISOR_PRESENT);
    }
    registers.rax = u64::from(result.eax);
    registers.rbx = u64::from(result.ebx);
    registers.rcx = u64::from(result.ecx);
    registers.rdx = u64::from(result.edx);
}
