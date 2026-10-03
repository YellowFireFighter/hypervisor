//! How a VMCS field is named, and the fields a hypervisor names.
//!
//! Every field of a VMCS is addressed by a 32-bit encoding rather than by an
//! offset, and the encoding is not arbitrary: it packs the four things a
//! caller has to know to use the field. Bit 0 is the *access type*, which
//! selects the low or high half of a 64-bit field. Bits 9:1 are an *index*
//! that distinguishes fields of the same kind and width. Bits 11:10 are the
//! *kind* — whether the field controls the guest, reports why it exited, holds
//! the guest's state, or holds the host's. Bits 14:13 are the *width*, which
//! says how many bytes `VMREAD` returns and `VMWRITE` takes.
//!
//! That the width is part of the name is the subtle part. A `VMREAD` of a
//! 32-bit field into a 64-bit register zero-extends; a `VMREAD` of a 16-bit
//! field reads two bytes. Pairing each field with the width its encoding
//! declares — rather than leaving the caller to remember it — is what keeps a
//! field from being read at the wrong size, which the processor does not
//! refuse.
//!
//! # High halves
//!
//! Only 64-bit fields have a high half, and most of the time a caller reads one
//! whole with a single 64-bit `VMREAD` of the full encoding. The high
//! encoding — the same field with bit 0 set — exists for a 32-bit host reaching
//! a 64-bit field in two halves, which pulzar does not do; it is modelled so
//! the encoding scheme is complete and so [`FieldEncoding::high`] can be stated
//! and tested, not because the fields below are listed twice.

use core::fmt::{self, Debug, Formatter};

/// How much a VMCS field holds, which is how much `VMREAD` returns and
/// `VMWRITE` takes.
///
/// The value is the width code the architecture puts in bits 14:13 of an
/// encoding, so it round-trips through [`FieldEncoding`] unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Width {
    /// Two bytes: a segment selector, an interrupt-status word, an index.
    Word = 0,
    /// Eight bytes: a physical address, a control-field bitmap pointer, a
    /// 64-bit model-specific register the guest runs with.
    QuadWord = 1,
    /// Four bytes: a control word, an exit reason, a segment limit or access
    /// rights.
    DoubleWord = 2,
    /// The width of a general register in the processor's current mode, which
    /// for the long-mode guests pulzar runs is eight bytes: a control register,
    /// a segment base, `RIP`, `RSP`, `RFLAGS`.
    Natural = 3,
}

impl Width {
    /// The width a width code names, or `None` for a value outside the two
    /// bits it comes from — which cannot happen for a code taken from an
    /// encoding, and is how a caller decoding a raw register learns it had a
    /// malformed one.
    #[must_use]
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Word),
            1 => Some(Self::QuadWord),
            2 => Some(Self::DoubleWord),
            3 => Some(Self::Natural),
            _ => None,
        }
    }

    /// The two-bit code for this width.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Which part of a guest a field belongs to, from bits 11:10 of an encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    /// A control field: what the processor does on the guest's behalf and what
    /// it intercepts. Written by the host, never by an exit.
    Control = 0,
    /// A field the processor fills in to say why the guest exited and what it
    /// was doing. Read-only to the host.
    ExitInformation = 1,
    /// The guest's own processor state, loaded on entry and saved on exit.
    GuestState = 2,
    /// The host's processor state, loaded when the guest exits.
    HostState = 3,
}

impl Kind {
    /// The kind a two-bit code names. Total, because both bits are always a
    /// valid kind.
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code & Self::MASK {
            0 => Self::Control,
            1 => Self::ExitInformation,
            2 => Self::GuestState,
            _ => Self::HostState,
        }
    }

    /// The two-bit code for this kind.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// The mask the code occupies before it is shifted into place.
    const MASK: u8 = 0b11;
}

/// Which half of a 64-bit field an encoding reaches, from bit 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Access {
    /// The whole field, or the low half of a 64-bit field read in two.
    Full = 0,
    /// The high half of a 64-bit field, valid only for [`Width::QuadWord`].
    High = 1,
}

/// A VMCS field's 32-bit encoding: the value `VMREAD` and `VMWRITE` name it by.
///
/// Comparisons against the named constants below are the intended way to test
/// one, and the accessors are the intended way to take one apart. The reserved
/// bits above the width are not modelled: an encoding is only ever built from a
/// constant here or read back from the processor, and no constructor lets a
/// caller set them.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct FieldEncoding(u32);

/// Bit holding the access type.
const ACCESS_SHIFT: u32 = 0;
/// Lowest bit of the index.
const INDEX_SHIFT: u32 = 1;
/// Lowest bit of the kind.
const KIND_SHIFT: u32 = 10;
/// Lowest bit of the width.
const WIDTH_SHIFT: u32 = 13;

/// Bits the index occupies once shifted down.
const INDEX_MASK: u32 = 0x1FF;
/// Bits the kind occupies once shifted down.
const KIND_MASK: u32 = 0b11;
/// Bits the width occupies once shifted down.
const WIDTH_MASK: u32 = 0b11;

impl FieldEncoding {
    /// The encoding with `access`, `index`, `kind` and `width`, which is how
    /// every constant below is built.
    #[must_use]
    const fn of(access: Access, index: u16, kind: Kind, width: Width) -> Self {
        Self(
            ((access as u32) << ACCESS_SHIFT)
                | ((index as u32 & INDEX_MASK) << INDEX_SHIFT)
                | ((kind.code() as u32) << KIND_SHIFT)
                | ((width.code() as u32) << WIDTH_SHIFT),
        )
    }

    /// The encoding the processor names this field by.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Whether this encoding reaches the whole field or its high half.
    #[must_use]
    pub const fn access(self) -> Access {
        if (self.0 >> ACCESS_SHIFT) & 1 == 0 {
            Access::Full
        } else {
            Access::High
        }
    }

    /// The index distinguishing this field from others of its kind and width.
    #[must_use]
    pub const fn index(self) -> u16 {
        ((self.0 >> INDEX_SHIFT) & INDEX_MASK) as u16
    }

    /// Which part of the guest this field belongs to.
    #[must_use]
    pub const fn kind(self) -> Kind {
        Kind::from_code(((self.0 >> KIND_SHIFT) & KIND_MASK) as u8)
    }

    /// How wide this field is.
    ///
    /// Total because the width bits are always one of the four widths; the
    /// `expect` cannot fire for a value whose two width bits are masked out of
    /// a `u32`.
    #[must_use]
    pub const fn width(self) -> Width {
        match Width::from_code(((self.0 >> WIDTH_SHIFT) & WIDTH_MASK) as u8) {
            Some(width) => width,
            None => unreachable!(),
        }
    }

    /// The encoding for the high half of this field.
    ///
    /// Meaningful only for a [`Width::QuadWord`] field; for any other it names
    /// an encoding the processor does not define, which is why this is used
    /// only where the width is already known to be 64 bits.
    #[must_use]
    pub const fn high(self) -> Self {
        Self(self.0 | (1 << ACCESS_SHIFT))
    }
}

impl Debug for FieldEncoding {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FieldEncoding")
            .field("bits", &format_args!("{:#06x}", self.0))
            .field("kind", &self.kind())
            .field("width", &self.width())
            .field("index", &self.index())
            .finish()
    }
}

/// A named VMCS field, paired with the encoding the processor knows it by.
///
/// Spelled as associated constants on [`FieldEncoding`] rather than as a
/// separate enum, for the reason `svm`'s exit codes are: the encoding *is* the
/// value, and a name that carried anything but the exact 32-bit encoding would
/// be a name the processor does not answer to. A caller writes
/// [`Field::GUEST_RIP`] where it means that field and never a bare number.
///
/// This holds the fields pulzar's VMX backend programs, across every kind and
/// width. It is not the whole of the architecture's table — the SGX,
/// shadow-VMCS and advanced-APIC fields a different hypervisor might touch are
/// not here — and adding one is adding a single constant, which the tests below
/// then check for the same self-consistency as the rest.
pub struct Field;

impl Field {
    /// Virtual-processor identifier, tagging this guest's cached translations.
    pub const VPID: FieldEncoding = FieldEncoding::of(Access::Full, 0, Kind::Control, Width::Word);
    /// Posted-interrupt notification vector.
    pub const POSTED_INTR_NOTIFICATION: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::Control, Width::Word);
    /// EPTP-switching index for the VM-function that switches EPT pointers.
    pub const EPTP_INDEX: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::Control, Width::Word);

    /// Guest `ES` selector.
    pub const GUEST_ES_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::GuestState, Width::Word);
    /// Guest `CS` selector.
    pub const GUEST_CS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::GuestState, Width::Word);
    /// Guest `SS` selector.
    pub const GUEST_SS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::GuestState, Width::Word);
    /// Guest `DS` selector.
    pub const GUEST_DS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::GuestState, Width::Word);
    /// Guest `FS` selector.
    pub const GUEST_FS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::GuestState, Width::Word);
    /// Guest `GS` selector.
    pub const GUEST_GS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::GuestState, Width::Word);
    /// Guest `LDTR` selector.
    pub const GUEST_LDTR_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::GuestState, Width::Word);
    /// Guest `TR` selector.
    pub const GUEST_TR_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::GuestState, Width::Word);
    /// Guest interrupt status, for virtual-interrupt delivery.
    pub const GUEST_INTERRUPT_STATUS: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::GuestState, Width::Word);
    /// Page-modification-log index.
    pub const GUEST_PML_INDEX: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::GuestState, Width::Word);

    /// Host `ES` selector.
    pub const HOST_ES_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::HostState, Width::Word);
    /// Host `CS` selector.
    pub const HOST_CS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::HostState, Width::Word);
    /// Host `SS` selector.
    pub const HOST_SS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::HostState, Width::Word);
    /// Host `DS` selector.
    pub const HOST_DS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::HostState, Width::Word);
    /// Host `FS` selector.
    pub const HOST_FS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::HostState, Width::Word);
    /// Host `GS` selector.
    pub const HOST_GS_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::HostState, Width::Word);
    /// Host `TR` selector.
    pub const HOST_TR_SELECTOR: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::HostState, Width::Word);

    /// Physical address of the I/O bitmap for ports `0x0000`–`0x7FFF`.
    pub const IO_BITMAP_A: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::Control, Width::QuadWord);
    /// Physical address of the I/O bitmap for ports `0x8000`–`0xFFFF`.
    pub const IO_BITMAP_B: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::Control, Width::QuadWord);
    /// Physical address of the model-specific-register permission bitmaps.
    pub const MSR_BITMAP: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::Control, Width::QuadWord);
    /// Physical address of the list of registers stored on VM exit.
    pub const VM_EXIT_MSR_STORE_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::Control, Width::QuadWord);
    /// Physical address of the list of registers loaded on VM exit.
    pub const VM_EXIT_MSR_LOAD_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::Control, Width::QuadWord);
    /// Physical address of the list of registers loaded on VM entry.
    pub const VM_ENTRY_MSR_LOAD_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::Control, Width::QuadWord);
    /// Timestamp-counter offset added to the guest's reading.
    pub const TSC_OFFSET: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::Control, Width::QuadWord);
    /// Physical address of the virtual-APIC page.
    pub const VIRTUAL_APIC_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::Control, Width::QuadWord);
    /// Physical address of the APIC-access page.
    pub const APIC_ACCESS_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 10, Kind::Control, Width::QuadWord);
    /// Physical address of the posted-interrupt descriptor.
    pub const POSTED_INTR_DESC_ADDR: FieldEncoding =
        FieldEncoding::of(Access::Full, 11, Kind::Control, Width::QuadWord);
    /// Extended-page-table pointer: the root of the second translation.
    pub const EPT_POINTER: FieldEncoding =
        FieldEncoding::of(Access::Full, 13, Kind::Control, Width::QuadWord);
    /// Timestamp-counter multiplier applied to the guest's reading.
    pub const TSC_MULTIPLIER: FieldEncoding =
        FieldEncoding::of(Access::Full, 25, Kind::Control, Width::QuadWord);

    /// Guest-physical address of a nested-paging fault, filled in on EPT exits.
    pub const GUEST_PHYSICAL_ADDRESS: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::ExitInformation, Width::QuadWord);

    /// Link pointer to a shadow VMCS, `!0` when there is none.
    pub const VMCS_LINK_POINTER: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::GuestState, Width::QuadWord);
    /// Guest `IA32_DEBUGCTL`.
    pub const GUEST_IA32_DEBUGCTL: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::GuestState, Width::QuadWord);
    /// Guest `IA32_PAT`, loaded when the entry controls ask for it.
    pub const GUEST_IA32_PAT: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::GuestState, Width::QuadWord);
    /// Guest `IA32_EFER`, loaded when the entry controls ask for it.
    pub const GUEST_IA32_EFER: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::GuestState, Width::QuadWord);

    /// Host `IA32_PAT`, loaded on exit when the exit controls ask for it.
    pub const HOST_IA32_PAT: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::HostState, Width::QuadWord);
    /// Host `IA32_EFER`, loaded on exit when the exit controls ask for it.
    pub const HOST_IA32_EFER: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::HostState, Width::QuadWord);

    /// Pin-based execution controls: what asynchronous events exit.
    pub const PIN_BASED_CONTROLS: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::Control, Width::DoubleWord);
    /// Primary processor-based execution controls.
    pub const PRIMARY_PROC_CONTROLS: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::Control, Width::DoubleWord);
    /// Bitmap of exceptions that exit rather than being delivered to the guest.
    pub const EXCEPTION_BITMAP: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::Control, Width::DoubleWord);
    /// Mask selecting which page-fault error-code bits are matched.
    pub const PAGE_FAULT_ERROR_CODE_MASK: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::Control, Width::DoubleWord);
    /// Value the masked page-fault error-code bits are matched against.
    pub const PAGE_FAULT_ERROR_CODE_MATCH: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::Control, Width::DoubleWord);
    /// Number of CR3-target values in use.
    pub const CR3_TARGET_COUNT: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::Control, Width::DoubleWord);
    /// Primary VM-exit controls: what the processor does when the guest exits.
    pub const PRIMARY_VM_EXIT_CONTROLS: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::Control, Width::DoubleWord);
    /// Number of registers stored on VM exit.
    pub const VM_EXIT_MSR_STORE_COUNT: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::Control, Width::DoubleWord);
    /// Number of registers loaded on VM exit.
    pub const VM_EXIT_MSR_LOAD_COUNT: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::Control, Width::DoubleWord);
    /// VM-entry controls: what the processor does when it enters the guest.
    pub const VM_ENTRY_CONTROLS: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::Control, Width::DoubleWord);
    /// Number of registers loaded on VM entry.
    pub const VM_ENTRY_MSR_LOAD_COUNT: FieldEncoding =
        FieldEncoding::of(Access::Full, 10, Kind::Control, Width::DoubleWord);
    /// The event to inject on the next VM entry.
    pub const VM_ENTRY_INTERRUPTION_INFO: FieldEncoding =
        FieldEncoding::of(Access::Full, 11, Kind::Control, Width::DoubleWord);
    /// The error code for an injected event that carries one.
    pub const VM_ENTRY_EXCEPTION_ERROR_CODE: FieldEncoding =
        FieldEncoding::of(Access::Full, 12, Kind::Control, Width::DoubleWord);
    /// Length of the instruction an injected software event follows.
    pub const VM_ENTRY_INSTRUCTION_LENGTH: FieldEncoding =
        FieldEncoding::of(Access::Full, 13, Kind::Control, Width::DoubleWord);
    /// Task-priority threshold below which a virtual interrupt exits.
    pub const TPR_THRESHOLD: FieldEncoding =
        FieldEncoding::of(Access::Full, 14, Kind::Control, Width::DoubleWord);
    /// Secondary processor-based execution controls.
    pub const SECONDARY_PROC_CONTROLS: FieldEncoding =
        FieldEncoding::of(Access::Full, 15, Kind::Control, Width::DoubleWord);

    /// Why the last `VMLAUNCH` or `VMRESUME` failed without entering the guest.
    pub const VM_INSTRUCTION_ERROR: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::ExitInformation, Width::DoubleWord);
    /// The basic reason the guest exited.
    pub const EXIT_REASON: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::ExitInformation, Width::DoubleWord);
    /// Information about an interruption that caused the exit.
    pub const VM_EXIT_INTERRUPTION_INFO: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::ExitInformation, Width::DoubleWord);
    /// The error code for an exit-causing interruption that carries one.
    pub const VM_EXIT_INTERRUPTION_ERROR_CODE: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::ExitInformation, Width::DoubleWord);
    /// Information about an event being delivered when the exit occurred.
    pub const IDT_VECTORING_INFO: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::ExitInformation, Width::DoubleWord);
    /// The error code for that event, when it carries one.
    pub const IDT_VECTORING_ERROR_CODE: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::ExitInformation, Width::DoubleWord);
    /// Length of the instruction the guest exited on.
    pub const VM_EXIT_INSTRUCTION_LENGTH: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::ExitInformation, Width::DoubleWord);
    /// Decoded detail of the instruction the guest exited on.
    pub const VM_EXIT_INSTRUCTION_INFO: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::ExitInformation, Width::DoubleWord);

    /// Guest `ES` segment limit.
    pub const GUEST_ES_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::GuestState, Width::DoubleWord);
    /// Guest `CS` segment limit.
    pub const GUEST_CS_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::GuestState, Width::DoubleWord);
    /// Guest `SS` segment limit.
    pub const GUEST_SS_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::GuestState, Width::DoubleWord);
    /// Guest `DS` segment limit.
    pub const GUEST_DS_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::GuestState, Width::DoubleWord);
    /// Guest `FS` segment limit.
    pub const GUEST_FS_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::GuestState, Width::DoubleWord);
    /// Guest `GS` segment limit.
    pub const GUEST_GS_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::GuestState, Width::DoubleWord);
    /// Guest `LDTR` limit.
    pub const GUEST_LDTR_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::GuestState, Width::DoubleWord);
    /// Guest `TR` limit.
    pub const GUEST_TR_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::GuestState, Width::DoubleWord);
    /// Guest `GDTR` limit.
    pub const GUEST_GDTR_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::GuestState, Width::DoubleWord);
    /// Guest `IDTR` limit.
    pub const GUEST_IDTR_LIMIT: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::GuestState, Width::DoubleWord);
    /// Guest `ES` access rights.
    pub const GUEST_ES_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 10, Kind::GuestState, Width::DoubleWord);
    /// Guest `CS` access rights.
    pub const GUEST_CS_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 11, Kind::GuestState, Width::DoubleWord);
    /// Guest `SS` access rights.
    pub const GUEST_SS_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 12, Kind::GuestState, Width::DoubleWord);
    /// Guest `DS` access rights.
    pub const GUEST_DS_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 13, Kind::GuestState, Width::DoubleWord);
    /// Guest `FS` access rights.
    pub const GUEST_FS_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 14, Kind::GuestState, Width::DoubleWord);
    /// Guest `GS` access rights.
    pub const GUEST_GS_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 15, Kind::GuestState, Width::DoubleWord);
    /// Guest `LDTR` access rights.
    pub const GUEST_LDTR_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 16, Kind::GuestState, Width::DoubleWord);
    /// Guest `TR` access rights.
    pub const GUEST_TR_ACCESS_RIGHTS: FieldEncoding =
        FieldEncoding::of(Access::Full, 17, Kind::GuestState, Width::DoubleWord);
    /// Guest interruptibility state: what is blocking interrupts right now.
    pub const GUEST_INTERRUPTIBILITY_STATE: FieldEncoding =
        FieldEncoding::of(Access::Full, 18, Kind::GuestState, Width::DoubleWord);
    /// Guest activity state: running, halted, shut down, or waiting for SIPI.
    pub const GUEST_ACTIVITY_STATE: FieldEncoding =
        FieldEncoding::of(Access::Full, 19, Kind::GuestState, Width::DoubleWord);
    /// Guest `IA32_SYSENTER_CS`.
    pub const GUEST_IA32_SYSENTER_CS: FieldEncoding =
        FieldEncoding::of(Access::Full, 21, Kind::GuestState, Width::DoubleWord);
    /// Remaining ticks of the VMX-preemption timer.
    pub const VMX_PREEMPTION_TIMER_VALUE: FieldEncoding =
        FieldEncoding::of(Access::Full, 23, Kind::GuestState, Width::DoubleWord);

    /// Host `IA32_SYSENTER_CS`.
    pub const HOST_IA32_SYSENTER_CS: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::HostState, Width::DoubleWord);

    /// Mask of `CR0` bits the host owns; a guest write to one exits.
    pub const CR0_GUEST_HOST_MASK: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::Control, Width::Natural);
    /// Mask of `CR4` bits the host owns.
    pub const CR4_GUEST_HOST_MASK: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::Control, Width::Natural);
    /// Value the guest reads for the host-owned `CR0` bits.
    pub const CR0_READ_SHADOW: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::Control, Width::Natural);
    /// Value the guest reads for the host-owned `CR4` bits.
    pub const CR4_READ_SHADOW: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::Control, Width::Natural);

    /// The qualification of the exit: the detail its reason points to.
    pub const EXIT_QUALIFICATION: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::ExitInformation, Width::Natural);
    /// Guest-linear address of a fault, where the exit reports one.
    pub const GUEST_LINEAR_ADDRESS: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::ExitInformation, Width::Natural);

    /// Guest `CR0`.
    pub const GUEST_CR0: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::GuestState, Width::Natural);
    /// Guest `CR3`.
    pub const GUEST_CR3: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::GuestState, Width::Natural);
    /// Guest `CR4`.
    pub const GUEST_CR4: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::GuestState, Width::Natural);
    /// Guest `ES` base.
    pub const GUEST_ES_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::GuestState, Width::Natural);
    /// Guest `CS` base.
    pub const GUEST_CS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::GuestState, Width::Natural);
    /// Guest `SS` base.
    pub const GUEST_SS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::GuestState, Width::Natural);
    /// Guest `DS` base.
    pub const GUEST_DS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::GuestState, Width::Natural);
    /// Guest `FS` base.
    pub const GUEST_FS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::GuestState, Width::Natural);
    /// Guest `GS` base.
    pub const GUEST_GS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::GuestState, Width::Natural);
    /// Guest `LDTR` base.
    pub const GUEST_LDTR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::GuestState, Width::Natural);
    /// Guest `TR` base.
    pub const GUEST_TR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 10, Kind::GuestState, Width::Natural);
    /// Guest `GDTR` base.
    pub const GUEST_GDTR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 11, Kind::GuestState, Width::Natural);
    /// Guest `IDTR` base.
    pub const GUEST_IDTR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 12, Kind::GuestState, Width::Natural);
    /// Guest `DR7`.
    pub const GUEST_DR7: FieldEncoding =
        FieldEncoding::of(Access::Full, 13, Kind::GuestState, Width::Natural);
    /// Guest `RSP`.
    pub const GUEST_RSP: FieldEncoding =
        FieldEncoding::of(Access::Full, 14, Kind::GuestState, Width::Natural);
    /// Guest `RIP`.
    pub const GUEST_RIP: FieldEncoding =
        FieldEncoding::of(Access::Full, 15, Kind::GuestState, Width::Natural);
    /// Guest `RFLAGS`.
    pub const GUEST_RFLAGS: FieldEncoding =
        FieldEncoding::of(Access::Full, 16, Kind::GuestState, Width::Natural);
    /// Guest pending debug exceptions.
    pub const GUEST_PENDING_DBG_EXCEPTIONS: FieldEncoding =
        FieldEncoding::of(Access::Full, 17, Kind::GuestState, Width::Natural);
    /// Guest `IA32_SYSENTER_ESP`.
    pub const GUEST_IA32_SYSENTER_ESP: FieldEncoding =
        FieldEncoding::of(Access::Full, 18, Kind::GuestState, Width::Natural);
    /// Guest `IA32_SYSENTER_EIP`.
    pub const GUEST_IA32_SYSENTER_EIP: FieldEncoding =
        FieldEncoding::of(Access::Full, 19, Kind::GuestState, Width::Natural);

    /// Host `CR0`.
    pub const HOST_CR0: FieldEncoding =
        FieldEncoding::of(Access::Full, 0, Kind::HostState, Width::Natural);
    /// Host `CR3`.
    pub const HOST_CR3: FieldEncoding =
        FieldEncoding::of(Access::Full, 1, Kind::HostState, Width::Natural);
    /// Host `CR4`.
    pub const HOST_CR4: FieldEncoding =
        FieldEncoding::of(Access::Full, 2, Kind::HostState, Width::Natural);
    /// Host `FS` base.
    pub const HOST_FS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 3, Kind::HostState, Width::Natural);
    /// Host `GS` base.
    pub const HOST_GS_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 4, Kind::HostState, Width::Natural);
    /// Host `TR` base.
    pub const HOST_TR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 5, Kind::HostState, Width::Natural);
    /// Host `GDTR` base.
    pub const HOST_GDTR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 6, Kind::HostState, Width::Natural);
    /// Host `IDTR` base.
    pub const HOST_IDTR_BASE: FieldEncoding =
        FieldEncoding::of(Access::Full, 7, Kind::HostState, Width::Natural);
    /// Host `IA32_SYSENTER_ESP`.
    pub const HOST_IA32_SYSENTER_ESP: FieldEncoding =
        FieldEncoding::of(Access::Full, 8, Kind::HostState, Width::Natural);
    /// Host `IA32_SYSENTER_EIP`.
    pub const HOST_IA32_SYSENTER_EIP: FieldEncoding =
        FieldEncoding::of(Access::Full, 9, Kind::HostState, Width::Natural);
    /// Host `RSP`.
    pub const HOST_RSP: FieldEncoding =
        FieldEncoding::of(Access::Full, 10, Kind::HostState, Width::Natural);
    /// Host `RIP`.
    pub const HOST_RIP: FieldEncoding =
        FieldEncoding::of(Access::Full, 11, Kind::HostState, Width::Natural);

    /// Every field above, for the tests that check the whole set at once.
    #[cfg(test)]
    const ALL: &'static [(FieldEncoding, u32)] = &[
        (Self::VPID, 0x0000),
        (Self::POSTED_INTR_NOTIFICATION, 0x0002),
        (Self::EPTP_INDEX, 0x0004),
        (Self::GUEST_ES_SELECTOR, 0x0800),
        (Self::GUEST_CS_SELECTOR, 0x0802),
        (Self::GUEST_SS_SELECTOR, 0x0804),
        (Self::GUEST_DS_SELECTOR, 0x0806),
        (Self::GUEST_FS_SELECTOR, 0x0808),
        (Self::GUEST_GS_SELECTOR, 0x080A),
        (Self::GUEST_LDTR_SELECTOR, 0x080C),
        (Self::GUEST_TR_SELECTOR, 0x080E),
        (Self::GUEST_INTERRUPT_STATUS, 0x0810),
        (Self::GUEST_PML_INDEX, 0x0812),
        (Self::HOST_ES_SELECTOR, 0x0C00),
        (Self::HOST_CS_SELECTOR, 0x0C02),
        (Self::HOST_SS_SELECTOR, 0x0C04),
        (Self::HOST_DS_SELECTOR, 0x0C06),
        (Self::HOST_FS_SELECTOR, 0x0C08),
        (Self::HOST_GS_SELECTOR, 0x0C0A),
        (Self::HOST_TR_SELECTOR, 0x0C0C),
        (Self::IO_BITMAP_A, 0x2000),
        (Self::IO_BITMAP_B, 0x2002),
        (Self::MSR_BITMAP, 0x2004),
        (Self::VM_EXIT_MSR_STORE_ADDR, 0x2006),
        (Self::VM_EXIT_MSR_LOAD_ADDR, 0x2008),
        (Self::VM_ENTRY_MSR_LOAD_ADDR, 0x200A),
        (Self::TSC_OFFSET, 0x2010),
        (Self::VIRTUAL_APIC_ADDR, 0x2012),
        (Self::APIC_ACCESS_ADDR, 0x2014),
        (Self::POSTED_INTR_DESC_ADDR, 0x2016),
        (Self::EPT_POINTER, 0x201A),
        (Self::TSC_MULTIPLIER, 0x2032),
        (Self::GUEST_PHYSICAL_ADDRESS, 0x2400),
        (Self::VMCS_LINK_POINTER, 0x2800),
        (Self::GUEST_IA32_DEBUGCTL, 0x2802),
        (Self::GUEST_IA32_PAT, 0x2804),
        (Self::GUEST_IA32_EFER, 0x2806),
        (Self::HOST_IA32_PAT, 0x2C00),
        (Self::HOST_IA32_EFER, 0x2C02),
        (Self::PIN_BASED_CONTROLS, 0x4000),
        (Self::PRIMARY_PROC_CONTROLS, 0x4002),
        (Self::EXCEPTION_BITMAP, 0x4004),
        (Self::PAGE_FAULT_ERROR_CODE_MASK, 0x4006),
        (Self::PAGE_FAULT_ERROR_CODE_MATCH, 0x4008),
        (Self::CR3_TARGET_COUNT, 0x400A),
        (Self::PRIMARY_VM_EXIT_CONTROLS, 0x400C),
        (Self::VM_EXIT_MSR_STORE_COUNT, 0x400E),
        (Self::VM_EXIT_MSR_LOAD_COUNT, 0x4010),
        (Self::VM_ENTRY_CONTROLS, 0x4012),
        (Self::VM_ENTRY_MSR_LOAD_COUNT, 0x4014),
        (Self::VM_ENTRY_INTERRUPTION_INFO, 0x4016),
        (Self::VM_ENTRY_EXCEPTION_ERROR_CODE, 0x4018),
        (Self::VM_ENTRY_INSTRUCTION_LENGTH, 0x401A),
        (Self::TPR_THRESHOLD, 0x401C),
        (Self::SECONDARY_PROC_CONTROLS, 0x401E),
        (Self::VM_INSTRUCTION_ERROR, 0x4400),
        (Self::EXIT_REASON, 0x4402),
        (Self::VM_EXIT_INTERRUPTION_INFO, 0x4404),
        (Self::VM_EXIT_INTERRUPTION_ERROR_CODE, 0x4406),
        (Self::IDT_VECTORING_INFO, 0x4408),
        (Self::IDT_VECTORING_ERROR_CODE, 0x440A),
        (Self::VM_EXIT_INSTRUCTION_LENGTH, 0x440C),
        (Self::VM_EXIT_INSTRUCTION_INFO, 0x440E),
        (Self::GUEST_ES_LIMIT, 0x4800),
        (Self::GUEST_CS_LIMIT, 0x4802),
        (Self::GUEST_SS_LIMIT, 0x4804),
        (Self::GUEST_DS_LIMIT, 0x4806),
        (Self::GUEST_FS_LIMIT, 0x4808),
        (Self::GUEST_GS_LIMIT, 0x480A),
        (Self::GUEST_LDTR_LIMIT, 0x480C),
        (Self::GUEST_TR_LIMIT, 0x480E),
        (Self::GUEST_GDTR_LIMIT, 0x4810),
        (Self::GUEST_IDTR_LIMIT, 0x4812),
        (Self::GUEST_ES_ACCESS_RIGHTS, 0x4814),
        (Self::GUEST_CS_ACCESS_RIGHTS, 0x4816),
        (Self::GUEST_SS_ACCESS_RIGHTS, 0x4818),
        (Self::GUEST_DS_ACCESS_RIGHTS, 0x481A),
        (Self::GUEST_FS_ACCESS_RIGHTS, 0x481C),
        (Self::GUEST_GS_ACCESS_RIGHTS, 0x481E),
        (Self::GUEST_LDTR_ACCESS_RIGHTS, 0x4820),
        (Self::GUEST_TR_ACCESS_RIGHTS, 0x4822),
        (Self::GUEST_INTERRUPTIBILITY_STATE, 0x4824),
        (Self::GUEST_ACTIVITY_STATE, 0x4826),
        (Self::GUEST_IA32_SYSENTER_CS, 0x482A),
        (Self::VMX_PREEMPTION_TIMER_VALUE, 0x482E),
        (Self::HOST_IA32_SYSENTER_CS, 0x4C00),
        (Self::CR0_GUEST_HOST_MASK, 0x6000),
        (Self::CR4_GUEST_HOST_MASK, 0x6002),
        (Self::CR0_READ_SHADOW, 0x6004),
        (Self::CR4_READ_SHADOW, 0x6006),
        (Self::EXIT_QUALIFICATION, 0x6400),
        (Self::GUEST_LINEAR_ADDRESS, 0x640A),
        (Self::GUEST_CR0, 0x6800),
        (Self::GUEST_CR3, 0x6802),
        (Self::GUEST_CR4, 0x6804),
        (Self::GUEST_ES_BASE, 0x6806),
        (Self::GUEST_CS_BASE, 0x6808),
        (Self::GUEST_SS_BASE, 0x680A),
        (Self::GUEST_DS_BASE, 0x680C),
        (Self::GUEST_FS_BASE, 0x680E),
        (Self::GUEST_GS_BASE, 0x6810),
        (Self::GUEST_LDTR_BASE, 0x6812),
        (Self::GUEST_TR_BASE, 0x6814),
        (Self::GUEST_GDTR_BASE, 0x6816),
        (Self::GUEST_IDTR_BASE, 0x6818),
        (Self::GUEST_DR7, 0x681A),
        (Self::GUEST_RSP, 0x681C),
        (Self::GUEST_RIP, 0x681E),
        (Self::GUEST_RFLAGS, 0x6820),
        (Self::GUEST_PENDING_DBG_EXCEPTIONS, 0x6822),
        (Self::GUEST_IA32_SYSENTER_ESP, 0x6824),
        (Self::GUEST_IA32_SYSENTER_EIP, 0x6826),
        (Self::HOST_CR0, 0x6C00),
        (Self::HOST_CR3, 0x6C02),
        (Self::HOST_CR4, 0x6C04),
        (Self::HOST_FS_BASE, 0x6C06),
        (Self::HOST_GS_BASE, 0x6C08),
        (Self::HOST_TR_BASE, 0x6C0A),
        (Self::HOST_GDTR_BASE, 0x6C0C),
        (Self::HOST_IDTR_BASE, 0x6C0E),
        (Self::HOST_IA32_SYSENTER_ESP, 0x6C10),
        (Self::HOST_IA32_SYSENTER_EIP, 0x6C12),
        (Self::HOST_RSP, 0x6C14),
        (Self::HOST_RIP, 0x6C16),
    ];
}

#[cfg(test)]
mod tests {
    use super::{Access, Field, FieldEncoding, Kind, Width};

    #[test]
    fn a_width_round_trips_through_its_code() {
        for width in [
            Width::Word,
            Width::QuadWord,
            Width::DoubleWord,
            Width::Natural,
        ] {
            assert_eq!(Width::from_code(width.code()), Some(width));
        }
        assert_eq!(Width::from_code(4), None);
    }

    #[test]
    fn a_kind_round_trips_through_its_code() {
        for kind in [
            Kind::Control,
            Kind::ExitInformation,
            Kind::GuestState,
            Kind::HostState,
        ] {
            assert_eq!(Kind::from_code(kind.code()), kind);
        }
    }

    #[test]
    fn the_accessors_recover_what_the_encoding_was_built_from() {
        let rip = Field::GUEST_RIP;
        assert_eq!(rip.access(), Access::Full);
        assert_eq!(rip.kind(), Kind::GuestState);
        assert_eq!(rip.width(), Width::Natural);
        assert_eq!(rip.index(), 15);
    }

    #[test]
    fn every_field_encodes_to_the_value_the_architecture_gives() {
        for &(field, bits) in Field::ALL {
            assert_eq!(
                field.bits(),
                bits,
                "{field:?} does not encode to {bits:#06x}"
            );
            // And the processor's raw value decodes back to the same parts,
            // which catches a transcribed constant whose bits say a different
            // kind or width than the group it was written in.
            let round = FieldEncoding(bits);
            assert_eq!(round.kind(), field.kind());
            assert_eq!(round.width(), field.width());
            assert_eq!(round.index(), field.index());
        }
    }

    #[test]
    fn no_two_fields_share_an_encoding() {
        let all = Field::ALL;
        for (position, &(_, bits)) in all.iter().enumerate() {
            for &(_, other) in &all[position + 1..] {
                assert_ne!(bits, other, "two fields share encoding {bits:#06x}");
            }
        }
    }

    #[test]
    fn the_high_half_sets_the_access_bit_on_a_sixty_four_bit_field() {
        let high = Field::IO_BITMAP_A.high();
        assert_eq!(high.access(), Access::High);
        assert_eq!(high.kind(), Field::IO_BITMAP_A.kind());
        assert_eq!(high.width(), Width::QuadWord);
        assert_eq!(high.index(), Field::IO_BITMAP_A.index());
        assert_eq!(high.bits(), Field::IO_BITMAP_A.bits() | 1);
    }
}
