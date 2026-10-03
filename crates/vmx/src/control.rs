//! What the processor does on a guest's behalf, and what it intercepts.
//!
//! This is the VMX counterpart to `svm`'s intercept bitmaps, and it works
//! differently in a way that matters. AMD lets a hypervisor set any intercept
//! bit it likes. Intel does not: every one of the four control words has a
//! capability register saying which of its bits are forced to one, which are
//! forced to zero, and which the hypervisor may choose — and a `VMLAUNCH` with
//! a control bit outside what the capability register allows fails outright
//! rather than running the guest. So a control word is never written as a bare
//! set of desired bits; it is the desired bits *reconciled* against the
//! capability register, which is what [`Capability::reconcile`] does.
//!
//! The bit positions are defined here as flag sets, and nothing here reads a
//! register or writes a VMCS: a caller brings the capability register's value
//! and the controls it wants, and this says what it must actually write.

use bitflags::bitflags;

/// `IA32_VMX_PINBASED_CTLS`: the capability register for the pin-based
/// controls.
pub const IA32_VMX_PINBASED_CTLS: u32 = 0x481;
/// `IA32_VMX_PROCBASED_CTLS`: the capability register for the primary
/// processor-based controls.
pub const IA32_VMX_PROCBASED_CTLS: u32 = 0x482;
/// `IA32_VMX_EXIT_CTLS`: the capability register for the VM-exit controls.
pub const IA32_VMX_EXIT_CTLS: u32 = 0x483;
/// `IA32_VMX_ENTRY_CTLS`: the capability register for the VM-entry controls.
pub const IA32_VMX_ENTRY_CTLS: u32 = 0x484;
/// `IA32_VMX_PROCBASED_CTLS2`: the capability register for the secondary
/// processor-based controls.
pub const IA32_VMX_PROCBASED_CTLS2: u32 = 0x48B;

/// `IA32_VMX_TRUE_PINBASED_CTLS`: the pin-based capabilities reporting only the
/// bits that are genuinely flexible, available when
/// [`VmxBasic::has_true_controls`](crate::VmxBasic::has_true_controls) is set.
pub const IA32_VMX_TRUE_PINBASED_CTLS: u32 = 0x48D;
/// `IA32_VMX_TRUE_PROCBASED_CTLS`: the flexible primary processor-based
/// capabilities.
pub const IA32_VMX_TRUE_PROCBASED_CTLS: u32 = 0x48E;
/// `IA32_VMX_TRUE_EXIT_CTLS`: the flexible VM-exit capabilities.
pub const IA32_VMX_TRUE_EXIT_CTLS: u32 = 0x48F;
/// `IA32_VMX_TRUE_ENTRY_CTLS`: the flexible VM-entry capabilities.
pub const IA32_VMX_TRUE_ENTRY_CTLS: u32 = 0x490;

/// A capability register for one of the control words.
///
/// Its low doubleword is the allowed-0 settings — a bit set here must be set in
/// the control word — and its high doubleword is the allowed-1 settings — a bit
/// clear here must be clear in the control word. A bit set in allowed-1 and
/// clear in allowed-0 is the hypervisor's to choose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capability(u64);

impl Capability {
    /// Takes the capability register's raw value.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// The bits that must be set, whatever the hypervisor wants.
    #[must_use]
    pub const fn allowed0(self) -> u32 {
        (self.0 & 0xFFFF_FFFF) as u32
    }

    /// The bits that may be set; any bit clear here must be clear in the
    /// control word.
    #[must_use]
    pub const fn allowed1(self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// The control word to write so that `desired` is honoured where the
    /// processor allows it: the forced-one bits are set, the forced-zero bits
    /// cleared, and every flexible bit left as `desired` asked.
    ///
    /// A bit the hypervisor wanted but the processor forbids is dropped rather
    /// than refused — a caller that needs a particular capability checks
    /// [`allows`](Self::allows) for it first — so the result is always a word
    /// the processor will accept.
    #[must_use]
    pub const fn reconcile(self, desired: u32) -> u32 {
        (desired | self.allowed0()) & self.allowed1()
    }

    /// Whether every bit of `wanted` is one the processor permits to be set.
    #[must_use]
    pub const fn allows(self, wanted: u32) -> bool {
        self.allowed1() & wanted == wanted
    }

    /// Whether any bit of `wanted` is one the processor forces to be set.
    #[must_use]
    pub const fn forces(self, wanted: u32) -> bool {
        self.allowed0() & wanted != 0
    }
}

bitflags! {
    /// Pin-based VM-execution controls: which asynchronous events exit.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PinBased: u32 {
        /// External interrupts cause VM exits.
        const EXTERNAL_INTERRUPT_EXITING = 1 << 0;
        /// Non-maskable interrupts cause VM exits.
        const NMI_EXITING = 1 << 3;
        /// NMIs are tracked as virtual, so one can be held pending and
        /// re-delivered rather than lost.
        const VIRTUAL_NMIS = 1 << 5;
        /// The VMX-preemption timer runs and its expiry exits.
        const ACTIVATE_PREEMPTION_TIMER = 1 << 6;
        /// Posted interrupts are processed.
        const PROCESS_POSTED_INTERRUPTS = 1 << 7;
    }
}

bitflags! {
    /// Primary processor-based VM-execution controls: which synchronous
    /// instructions and events exit, and which secondary mechanisms are on.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PrimaryProc: u32 {
        /// Exit when an interrupt may be delivered to the guest.
        const INTERRUPT_WINDOW_EXITING = 1 << 2;
        /// Apply the timestamp-counter offset and multiplier to the guest.
        const USE_TSC_OFFSETTING = 1 << 3;
        /// `HLT` exits.
        const HLT_EXITING = 1 << 7;
        /// `INVLPG` exits.
        const INVLPG_EXITING = 1 << 9;
        /// `MWAIT` exits.
        const MWAIT_EXITING = 1 << 10;
        /// `RDPMC` exits.
        const RDPMC_EXITING = 1 << 11;
        /// `RDTSC` and `RDTSCP` exit.
        const RDTSC_EXITING = 1 << 12;
        /// Loads of `CR3` exit.
        const CR3_LOAD_EXITING = 1 << 15;
        /// Stores from `CR3` exit.
        const CR3_STORE_EXITING = 1 << 16;
        /// Loads of `CR8` exit.
        const CR8_LOAD_EXITING = 1 << 19;
        /// Stores from `CR8` exit.
        const CR8_STORE_EXITING = 1 << 20;
        /// The virtual-APIC page shadows `CR8`/the task-priority register.
        const USE_TPR_SHADOW = 1 << 21;
        /// Exit when an NMI may be delivered to the guest.
        const NMI_WINDOW_EXITING = 1 << 22;
        /// `MOV` to or from a debug register exits.
        const MOV_DR_EXITING = 1 << 23;
        /// Every I/O instruction exits.
        const UNCONDITIONAL_IO_EXITING = 1 << 24;
        /// I/O instructions exit according to the I/O bitmaps.
        const USE_IO_BITMAPS = 1 << 25;
        /// The monitor trap flag single-steps the guest.
        const MONITOR_TRAP_FLAG = 1 << 27;
        /// Model-specific-register accesses exit according to the MSR bitmaps.
        const USE_MSR_BITMAPS = 1 << 28;
        /// `MONITOR` exits.
        const MONITOR_EXITING = 1 << 29;
        /// `PAUSE` exits.
        const PAUSE_EXITING = 1 << 30;
        /// The secondary controls word is active.
        const ACTIVATE_SECONDARY_CONTROLS = 1 << 31;
    }
}

bitflags! {
    /// Secondary processor-based VM-execution controls, active only when
    /// [`PrimaryProc::ACTIVATE_SECONDARY_CONTROLS`] is set.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SecondaryProc: u32 {
        /// Accesses to the APIC-access page are virtualized.
        const VIRTUALIZE_APIC_ACCESSES = 1 << 0;
        /// Extended page tables provide the second translation.
        const ENABLE_EPT = 1 << 1;
        /// `LGDT`/`LIDT`/`LLDT`/`LTR` and their stores exit.
        const DESCRIPTOR_TABLE_EXITING = 1 << 2;
        /// `RDTSCP` is allowed in the guest rather than faulting.
        const ENABLE_RDTSCP = 1 << 3;
        /// x2APIC model-specific registers are virtualized.
        const VIRTUALIZE_X2APIC_MODE = 1 << 4;
        /// Cached translations are tagged by virtual-processor identifier.
        const ENABLE_VPID = 1 << 5;
        /// `WBINVD` exits.
        const WBINVD_EXITING = 1 << 6;
        /// The guest may run with no paging and no protected mode.
        const UNRESTRICTED_GUEST = 1 << 7;
        /// Reads of some APIC registers are virtualized.
        const APIC_REGISTER_VIRTUALIZATION = 1 << 8;
        /// Interrupts are delivered through the virtual-APIC page.
        const VIRTUAL_INTERRUPT_DELIVERY = 1 << 9;
        /// A run of `PAUSE` instructions exits.
        const PAUSE_LOOP_EXITING = 1 << 10;
        /// `RDRAND` exits.
        const RDRAND_EXITING = 1 << 11;
        /// `INVPCID` is allowed in the guest.
        const ENABLE_INVPCID = 1 << 12;
        /// `VMFUNC` is allowed in the guest.
        const ENABLE_VM_FUNCTIONS = 1 << 13;
        /// `VMREAD`/`VMWRITE` in the guest go to a shadow VMCS.
        const VMCS_SHADOWING = 1 << 14;
        /// `ENCLS` exits according to its bitmap.
        const ENABLE_ENCLS_EXITING = 1 << 15;
        /// `RDSEED` exits.
        const RDSEED_EXITING = 1 << 16;
        /// The page-modification log records guest writes.
        const ENABLE_PML = 1 << 17;
        /// An EPT violation raises a virtualization exception in the guest
        /// rather than exiting.
        const EPT_VIOLATION_VE = 1 << 18;
        /// VMX activity is concealed from Intel Processor Trace.
        const CONCEAL_VMX_FROM_PT = 1 << 19;
        /// `XSAVES`/`XRSTORS` are allowed in the guest.
        const ENABLE_XSAVES = 1 << 20;
        /// Execute permission in EPT depends on whether the access is
        /// supervisor or user.
        const MODE_BASED_EXECUTE_CONTROL_EPT = 1 << 22;
        /// The timestamp counter is scaled by the multiplier field.
        const USE_TSC_SCALING = 1 << 25;
    }
}

bitflags! {
    /// VM-exit controls: what the processor does when the guest exits.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VmExit: u32 {
        /// Save the guest's debug controls on exit.
        const SAVE_DEBUG_CONTROLS = 1 << 2;
        /// The host runs in 64-bit mode after the exit.
        const HOST_ADDRESS_SPACE_SIZE = 1 << 9;
        /// Load the host `IA32_PERF_GLOBAL_CTRL` on exit.
        const LOAD_IA32_PERF_GLOBAL_CTRL = 1 << 12;
        /// Acknowledge an exit-causing external interrupt, reading its vector
        /// into the exit interruption-information field.
        const ACKNOWLEDGE_INTERRUPT_ON_EXIT = 1 << 15;
        /// Save the guest `IA32_PAT` on exit.
        const SAVE_IA32_PAT = 1 << 18;
        /// Load the host `IA32_PAT` on exit.
        const LOAD_IA32_PAT = 1 << 19;
        /// Save the guest `IA32_EFER` on exit.
        const SAVE_IA32_EFER = 1 << 20;
        /// Load the host `IA32_EFER` on exit.
        const LOAD_IA32_EFER = 1 << 21;
        /// Save the VMX-preemption timer value on exit.
        const SAVE_PREEMPTION_TIMER_VALUE = 1 << 22;
        /// VMX activity is concealed from Intel Processor Trace across the exit.
        const CONCEAL_VMX_FROM_PT = 1 << 24;
    }
}

bitflags! {
    /// VM-entry controls: what the processor does when it enters the guest.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VmEntry: u32 {
        /// Load the guest's debug controls on entry.
        const LOAD_DEBUG_CONTROLS = 1 << 2;
        /// The guest runs in 64-bit mode.
        const IA32E_MODE_GUEST = 1 << 9;
        /// Entry is to system-management mode.
        const ENTRY_TO_SMM = 1 << 10;
        /// Deactivate the dual-monitor treatment of SMM.
        const DEACTIVATE_DUAL_MONITOR = 1 << 11;
        /// Load the guest `IA32_PERF_GLOBAL_CTRL` on entry.
        const LOAD_IA32_PERF_GLOBAL_CTRL = 1 << 13;
        /// Load the guest `IA32_PAT` on entry.
        const LOAD_IA32_PAT = 1 << 14;
        /// Load the guest `IA32_EFER` on entry.
        const LOAD_IA32_EFER = 1 << 15;
        /// VMX activity is concealed from Intel Processor Trace across entry.
        const CONCEAL_VMX_FROM_PT = 1 << 17;
    }
}

#[cfg(test)]
mod tests {
    use super::{Capability, PrimaryProc, SecondaryProc};

    #[test]
    fn reconcile_forces_the_ones_sets_the_flexible_and_clears_the_forbidden() {
        // allowed-0 forces bit 0 set; allowed-1 permits bits 0, 1 and 2 but
        // not bit 3.
        let allowed0: u64 = 0b0001;
        let allowed1: u64 = 0b0111;
        let capability = Capability::from_bits(allowed0 | (allowed1 << 32));

        // Want bits 1 and 3. Bit 0 is forced on, bit 1 is honoured, bit 3 is
        // forbidden and dropped.
        assert_eq!(capability.reconcile(0b1010), 0b0011);
        // Wanting nothing still yields the forced bit.
        assert_eq!(capability.reconcile(0), 0b0001);
    }

    #[test]
    fn allows_and_forces_read_the_two_halves() {
        let capability = Capability::from_bits(0b0001 | (0b0111_u64 << 32));
        assert!(capability.allows(0b0110));
        assert!(!capability.allows(0b1000));
        assert!(capability.forces(0b0001));
        assert!(!capability.forces(0b0110));
    }

    #[test]
    fn the_secondary_control_switch_is_the_top_primary_bit() {
        assert_eq!(PrimaryProc::ACTIVATE_SECONDARY_CONTROLS.bits(), 1 << 31);
        assert_eq!(SecondaryProc::ENABLE_EPT.bits(), 1 << 1);
    }
}
