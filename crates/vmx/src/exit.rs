//! Why a guest stopped.
//!
//! The VMX counterpart to `svm`'s exit codes. The processor writes a 32-bit
//! exit-reason field on every exit: its low sixteen bits are the *basic* reason
//! — a small dense enumeration, which is what a hypervisor dispatches on — and
//! the bits above carry flags about the exit, the one that matters being
//! whether the exit was a VM-entry *failure* rather than something the guest
//! did. An entry failure means no guest instruction executed and the state the
//! processor rejected is in the VMCS, which is the VMX equivalent of `svm`'s
//! `INVALID` exit: the thing to recognize before looking for a fault in a guest
//! that never ran.

use core::fmt::{self, Debug, Display, Formatter};

/// The low-sixteen-bit basic exit reason.
///
/// Comparisons against the named constants are the intended way to test one.
/// The numbering is the architecture's, so a reason the processor reports that
/// this build has no name for still round-trips as its number.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct BasicExitReason(u16);

impl BasicExitReason {
    /// A hardware exception or non-maskable interrupt.
    pub const EXCEPTION_OR_NMI: Self = Self(0);
    /// A maskable external interrupt.
    pub const EXTERNAL_INTERRUPT: Self = Self(1);
    /// A triple fault: a fault while delivering a fault that itself faulted.
    pub const TRIPLE_FAULT: Self = Self(2);
    /// An `INIT` signal arrived.
    pub const INIT_SIGNAL: Self = Self(3);
    /// A start-up inter-processor interrupt arrived.
    pub const STARTUP_IPI: Self = Self(4);
    /// An interrupt window opened: the guest can take an interrupt now.
    pub const INTERRUPT_WINDOW: Self = Self(7);
    /// An NMI window opened: the guest can take an NMI now.
    pub const NMI_WINDOW: Self = Self(8);
    /// A task switch.
    pub const TASK_SWITCH: Self = Self(9);
    /// `CPUID`.
    pub const CPUID: Self = Self(10);
    /// `HLT`.
    pub const HLT: Self = Self(12);
    /// `INVD`.
    pub const INVD: Self = Self(13);
    /// `INVLPG`.
    pub const INVLPG: Self = Self(14);
    /// `RDPMC`.
    pub const RDPMC: Self = Self(15);
    /// `RDTSC`.
    pub const RDTSC: Self = Self(16);
    /// `VMCALL`, the guest-to-hypervisor call — the VMX `VMMCALL`.
    pub const VMCALL: Self = Self(18);
    /// `VMCLEAR`.
    pub const VMCLEAR: Self = Self(19);
    /// `VMLAUNCH`.
    pub const VMLAUNCH: Self = Self(20);
    /// `VMPTRLD`.
    pub const VMPTRLD: Self = Self(21);
    /// `VMPTRST`.
    pub const VMPTRST: Self = Self(22);
    /// `VMREAD`.
    pub const VMREAD: Self = Self(23);
    /// `VMRESUME`.
    pub const VMRESUME: Self = Self(24);
    /// `VMWRITE`.
    pub const VMWRITE: Self = Self(25);
    /// `VMXOFF`.
    pub const VMXOFF: Self = Self(26);
    /// `VMXON`.
    pub const VMXON: Self = Self(27);
    /// A `MOV` to or from a control register.
    pub const CONTROL_REGISTER_ACCESS: Self = Self(28);
    /// A `MOV` to or from a debug register.
    pub const MOV_DR: Self = Self(29);
    /// An I/O instruction.
    pub const IO_INSTRUCTION: Self = Self(30);
    /// `RDMSR`.
    pub const RDMSR: Self = Self(31);
    /// `WRMSR`.
    pub const WRMSR: Self = Self(32);
    /// VM entry failed because the guest state was invalid.
    pub const ENTRY_FAILURE_GUEST_STATE: Self = Self(33);
    /// VM entry failed while loading model-specific registers.
    pub const ENTRY_FAILURE_MSR_LOADING: Self = Self(34);
    /// `MWAIT`.
    pub const MWAIT: Self = Self(36);
    /// The monitor trap flag fired.
    pub const MONITOR_TRAP_FLAG: Self = Self(37);
    /// `MONITOR`.
    pub const MONITOR: Self = Self(39);
    /// `PAUSE`.
    pub const PAUSE: Self = Self(40);
    /// VM entry failed because of a machine check.
    pub const ENTRY_FAILURE_MACHINE_CHECK: Self = Self(41);
    /// The virtual task priority dropped below its threshold.
    pub const TPR_BELOW_THRESHOLD: Self = Self(43);
    /// An access to the APIC-access page.
    pub const APIC_ACCESS: Self = Self(44);
    /// A virtualized end-of-interrupt.
    pub const VIRTUALIZED_EOI: Self = Self(45);
    /// An access to `GDTR` or `IDTR`.
    pub const GDTR_IDTR_ACCESS: Self = Self(46);
    /// An access to `LDTR` or `TR`.
    pub const LDTR_TR_ACCESS: Self = Self(47);
    /// An EPT violation: a guest-physical access the EPT entries forbade.
    pub const EPT_VIOLATION: Self = Self(48);
    /// An EPT misconfiguration: a malformed EPT entry on a guest access.
    pub const EPT_MISCONFIGURATION: Self = Self(49);
    /// `INVEPT`.
    pub const INVEPT: Self = Self(50);
    /// `RDTSCP`.
    pub const RDTSCP: Self = Self(51);
    /// The VMX-preemption timer reached zero.
    pub const PREEMPTION_TIMER_EXPIRED: Self = Self(52);
    /// `INVVPID`.
    pub const INVVPID: Self = Self(53);
    /// `WBINVD`.
    pub const WBINVD: Self = Self(54);
    /// `XSETBV`.
    pub const XSETBV: Self = Self(55);
    /// A write to an APIC register that must be completed by the hypervisor.
    pub const APIC_WRITE: Self = Self(56);
    /// `RDRAND`.
    pub const RDRAND: Self = Self(57);
    /// `INVPCID`.
    pub const INVPCID: Self = Self(58);
    /// `VMFUNC`.
    pub const VMFUNC: Self = Self(59);
    /// `ENCLS`.
    pub const ENCLS: Self = Self(60);
    /// `RDSEED`.
    pub const RDSEED: Self = Self(61);
    /// The page-modification log filled.
    pub const PML_FULL: Self = Self(62);
    /// `XSAVES`.
    pub const XSAVES: Self = Self(63);
    /// `XRSTORS`.
    pub const XRSTORS: Self = Self(64);

    /// The number this reason is reported as.
    #[must_use]
    pub const fn number(self) -> u16 {
        self.0
    }
}

impl Debug for BasicExitReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "BasicExitReason({})", self.0)
    }
}

impl Display for BasicExitReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(formatter, "exit reason {}", self.0)
    }
}

/// The whole 32-bit exit-reason field.
///
/// The basic reason plus the flags above it, of which the one a hypervisor
/// acts on is [`entry_failed`](Self::entry_failed).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct ExitReason(u32);

/// The basic reason is the low sixteen bits.
const BASIC_MASK: u32 = 0xFFFF;
/// Bit 31 is set when the exit was a failed VM entry rather than a guest
/// action.
const ENTRY_FAILURE_BIT: u32 = 1 << 31;

impl ExitReason {
    /// Takes the exit-reason field's raw value.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The basic reason the hypervisor dispatches on.
    #[must_use]
    pub const fn basic(self) -> BasicExitReason {
        BasicExitReason((self.0 & BASIC_MASK) as u16)
    }

    /// Whether this exit is a failed VM entry — no guest instruction ran, and
    /// the rejected state is in the VMCS.
    #[must_use]
    pub const fn entry_failed(self) -> bool {
        self.0 & ENTRY_FAILURE_BIT != 0
    }
}

impl Debug for ExitReason {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExitReason")
            .field("basic", &self.basic())
            .field("entry_failed", &self.entry_failed())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{BasicExitReason, ExitReason};

    #[test]
    fn the_basic_reason_is_the_low_sixteen_bits() {
        let reason = ExitReason::from_bits(0x0000_000A);
        assert_eq!(reason.basic(), BasicExitReason::CPUID);
        assert!(!reason.entry_failed());
    }

    #[test]
    fn a_failed_entry_keeps_its_basic_reason_and_sets_the_flag() {
        let reason = ExitReason::from_bits(0x8000_0021);
        assert_eq!(reason.basic(), BasicExitReason::ENTRY_FAILURE_GUEST_STATE);
        assert!(reason.entry_failed());
    }

    #[test]
    fn the_named_reasons_carry_the_architectural_numbers() {
        assert_eq!(BasicExitReason::EPT_VIOLATION.number(), 48);
        assert_eq!(BasicExitReason::VMCALL.number(), 18);
        assert_eq!(BasicExitReason::CONTROL_REGISTER_ACCESS.number(), 28);
    }
}
