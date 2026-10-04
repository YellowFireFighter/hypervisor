//! The world switch: entering a guest and coming back.
//!
//! The counterpart to `vcpu`'s `switch`, and the piece of the Intel path whose
//! correctness is in register discipline that no review fully settles. VMX
//! saves and restores almost nothing of the general registers itself — only
//! `RSP` and `RIP`, which live in the VMCS — so this loads the guest's
//! registers before entry and saves them on exit by hand, around `VMLAUNCH` or
//! `VMRESUME`. It has launched a guest on an Intel processor, brought control
//! back cleanly through the exit path below with the guest's registers saved,
//! and taken the resume path too — a guest has been resumed across several
//! exits.
//!
//! # How control comes back
//!
//! Unlike `VMRUN`, which returns to the instruction after it, a VM exit resumes
//! the processor at the host `RIP` and host `RSP` recorded in the VMCS. So the
//! loop writes those two fields itself, just before entering: host `RSP` to the
//! current stack, host `RIP` to a label inside the asm. A VM exit lands on that
//! label with the stack as it was left, where the guest registers are saved and
//! the host's restored.
//!
//! # The register the context travels in
//!
//! Every value the loop needs across the entry — where the guest registers are,
//! whether to launch or resume, where to record a failure — is reached through
//! one pointer, pinned to `RCX`. It is pushed before entry so a VM exit, which
//! clobbers every register with the guest's, can recover it from the stack. The
//! callee-saved registers the asm uses are preserved by hand; `RBP` among them,
//! because it may be the frame pointer and cannot be named as a clobber.

use core::mem::offset_of;

use vmx::Field;

use crate::{Vmcs, error::VmFail};

/// The guest's general registers, plus what the world switch needs to carry
/// across an entry.
///
/// `RSP` is absent: it is the VMCS's `GUEST_RSP`, saved and restored by the
/// processor. The layout is `#[repr(C)]` so the offsets the asm is built on are
/// fixed, and they are asserted below.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Registers {
    /// Guest `RAX`.
    pub rax: u64,
    /// Guest `RBX`.
    pub rbx: u64,
    /// Guest `RCX`.
    pub rcx: u64,
    /// Guest `RDX`.
    pub rdx: u64,
    /// Guest `RSI`.
    pub rsi: u64,
    /// Guest `RDI`.
    pub rdi: u64,
    /// Guest `RBP`.
    pub rbp: u64,
    /// Guest `R8`.
    pub r8: u64,
    /// Guest `R9`.
    pub r9: u64,
    /// Guest `R10`.
    pub r10: u64,
    /// Guest `R11`.
    pub r11: u64,
    /// Guest `R12`.
    pub r12: u64,
    /// Guest `R13`.
    pub r13: u64,
    /// Guest `R14`.
    pub r14: u64,
    /// Guest `R15`.
    pub r15: u64,
    /// Zero to launch, nonzero to resume. Set from the VMCS's launch state.
    launched: u64,
    /// Written by the world switch: zero when the guest ran and exited,
    /// nonzero (the failed entry's `RFLAGS`) when `VMLAUNCH`/`VMRESUME` was
    /// rejected without entering the guest.
    entry_failed: u64,
}

/// How a guest entry ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entered {
    /// The guest ran and exited; read the exit reason from the VMCS.
    Exited,
    /// Entry was rejected without running the guest; the VM-instruction-error
    /// field holds why.
    Failed(VmFail),
}

/// Enters the guest the current VMCS describes, carrying `registers`, and
/// returns how the entry ended.
///
/// On [`Entered::Exited`] the guest's registers are back in `registers` and the
/// exit reason is in the VMCS. On [`Entered::Failed`] the guest did not run.
///
/// # Safety
///
/// `cell` must be the current VMCS on this processor, fully programmed — host
/// state, guest state and controls all written — and this processor in VMX
/// operation. The host state the processor restores on exit (segments, tables,
/// control registers) must match the environment this returns into, which it
/// does when [`crate::host::program`] captured the running host.
#[expect(
    clippy::too_many_lines,
    reason = "the world switch is one indivisible asm block; splitting it would hide the entry/exit register discipline that must be read as a whole"
)]
pub unsafe fn run(cell: &mut Vmcs, registers: &mut Registers) -> Entered {
    registers.launched = u64::from(cell.launched());
    registers.entry_failed = 0;

    // SAFETY: the caller guarantees a fully programmed current VMCS in VMX
    // operation; the asm preserves every callee-saved register it uses, writes
    // host RSP/RIP so a VM exit returns to the label below, and touches only
    // `registers` through the pinned pointer.
    unsafe {
        core::arch::asm!(
            "push rbp",
            "push rbx",
            "push r12",
            "push r13",
            "push r14",
            "push r15",
            "push rcx",                       // context pointer; HOST_RSP points here

            "mov rax, {HOST_RSP}",
            "vmwrite rax, rsp",
            "lea rax, [rip + 2f]",
            "mov rdx, {HOST_RIP}",
            "vmwrite rdx, rax",

            "mov rax, [rcx + {RAX}]",
            "mov rbx, [rcx + {RBX}]",
            "mov rdx, [rcx + {RDX}]",
            "mov rsi, [rcx + {RSI}]",
            "mov rdi, [rcx + {RDI}]",
            "mov rbp, [rcx + {RBP}]",
            "mov r8,  [rcx + {R8}]",
            "mov r9,  [rcx + {R9}]",
            "mov r10, [rcx + {R10}]",
            "mov r11, [rcx + {R11}]",
            "mov r12, [rcx + {R12}]",
            "mov r13, [rcx + {R13}]",
            "mov r14, [rcx + {R14}]",
            "mov r15, [rcx + {R15}]",
            "cmp qword ptr [rcx + {LAUNCHED}], 0",
            "mov rcx, [rcx + {RCX}]",         // guest RCX last; context now only on the stack

            "jne 3f",
            "vmlaunch",
            "jmp 4f",
            "3:",
            "vmresume",
            "4:",                             // entry failed: control never left for the guest
            "mov rcx, [rsp]",                 // recover the context pointer
            "pushfq",
            "pop qword ptr [rcx + {FAILED}]", // record the failing RFLAGS
            "jmp 5f",

            "2:",                             // VM exit lands here
            "push rcx",                       // free RCX; [rsp] = guest RCX, [rsp+8] = context
            "mov rcx, [rsp + 8]",
            "mov [rcx + {RAX}], rax",
            "mov [rcx + {RBX}], rbx",
            "mov [rcx + {RDX}], rdx",
            "mov [rcx + {RSI}], rsi",
            "mov [rcx + {RDI}], rdi",
            "mov [rcx + {RBP}], rbp",
            "mov [rcx + {R8}],  r8",
            "mov [rcx + {R9}],  r9",
            "mov [rcx + {R10}], r10",
            "mov [rcx + {R11}], r11",
            "mov [rcx + {R12}], r12",
            "mov [rcx + {R13}], r13",
            "mov [rcx + {R14}], r14",
            "mov [rcx + {R15}], r15",
            "mov rax, [rsp]",                 // guest RCX saved on the stack
            "mov [rcx + {RCX}], rax",
            "mov qword ptr [rcx + {FAILED}], 0",
            "pop rax",                        // discard the saved guest RCX

            "5:",
            "pop rcx",                        // discard the context pointer
            "pop r15",
            "pop r14",
            "pop r13",
            "pop r12",
            "pop rbx",
            "pop rbp",

            HOST_RSP = const Field::HOST_RSP.bits(),
            HOST_RIP = const Field::HOST_RIP.bits(),
            RAX = const offset_of!(Registers, rax),
            RBX = const offset_of!(Registers, rbx),
            RCX = const offset_of!(Registers, rcx),
            RDX = const offset_of!(Registers, rdx),
            RSI = const offset_of!(Registers, rsi),
            RDI = const offset_of!(Registers, rdi),
            RBP = const offset_of!(Registers, rbp),
            R8 = const offset_of!(Registers, r8),
            R9 = const offset_of!(Registers, r9),
            R10 = const offset_of!(Registers, r10),
            R11 = const offset_of!(Registers, r11),
            R12 = const offset_of!(Registers, r12),
            R13 = const offset_of!(Registers, r13),
            R14 = const offset_of!(Registers, r14),
            R15 = const offset_of!(Registers, r15),
            LAUNCHED = const offset_of!(Registers, launched),
            FAILED = const offset_of!(Registers, entry_failed),
            inout("rcx") core::ptr::from_mut(registers) => _,
            out("rax") _,
            out("rdx") _,
            out("rsi") _,
            out("rdi") _,
            out("r8") _,
            out("r9") _,
            out("r10") _,
            out("r11") _,
        );
    }

    if registers.entry_failed == 0 {
        cell.mark_launched();
        Entered::Exited
    } else {
        // The failing flags distinguish the two VMX failure modes, exactly as
        // the instruction wrappers decode them.
        Entered::Failed(
            match crate::error::Outcome::from_flags(registers.entry_failed) {
                crate::error::Outcome::FailValid => VmFail::Valid,
                _ => VmFail::Invalid,
            },
        )
    }
}

// The eight general registers VMX would clobber but that this asm declares as
// outputs must leave `RSP` alone, and the context register `RCX` is the one the
// guest's value is loaded into last; the offsets the asm reaches the context
// through are the struct's own, which these fix.
const _: () = assert!(
    offset_of!(Registers, rax) == 0,
    "the register block must start at RAX, which the asm addresses as offset zero",
);
const _: () = assert!(
    offset_of!(Registers, entry_failed) == 16 * size_of::<u64>(),
    "the two control words must follow the fifteen saved registers with no gap",
);
