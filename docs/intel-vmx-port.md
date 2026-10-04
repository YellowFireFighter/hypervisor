# Intel VMX port — status and roadmap

pulzar is an AMD SVM hypervisor. An Intel VMX backend is being built alongside
the SVM one so a single image can eventually run on either vendor. The
foundational layers exist and are proven on Intel hardware; the higher layers
that a real OS guest needs do not yet exist. This file is the map of what is
done, how it is tested, and what remains.

## The crates, and their SVM counterparts

| Intel crate | Role | SVM counterpart |
|---|---|---|
| `vmx` | Definitions only: VMCS field encodings, pin/proc/exit/entry control bits + capability reconciliation, exit reasons, event-injection format, segment access-rights, VMXON/VMCS region headers, EPT entry and pointer formats, `IA32_VMX_BASIC`, VMX support + feature-control decoding. No `unsafe`, no hardware. | `svm` |
| `ept` | EPT tree builder over a `Memory` trait: `identity()` (2-MiB large-page identity map) and `map()` (one 4-KiB page). | `npt` |
| `vmcs` | Executable layer: `enable` (`VMXON`), `instr` (instruction wrappers), `vmcs` (VMCS lifecycle + typed field access), `run` (the `VMLAUNCH`/`VMRESUME` world switch), `host`/`guest`/`controls` (VMCS state programming), `fixed` (CR0/CR4 fixed-bit reconcile), `error` (`Outcome`/`VmFail`). | `vcpu` |
| `vmexits` | Exit dispatch: `dispatch()` answers each exit and returns `Flow` (resume / vmcall / stop). Handles `CPUID` (forward the machine's answer), `RDMSR`/`WRMSR` (forward through `probe`), `HLT`, `VMCALL`, triple fault; decodes the control-register and EPT-violation exit qualifications (`control_register`, `violation`) and stops on them, since their handlers are later layers. | `exits` |
| `crates/hv-core/src/selftest.rs` | Feature-gated (`vmx-selftest`) battery that drives all of the above on real hardware and reports `[PASS]`/`[FAIL]` per check. | — |

## Verification status

Host unit tests (`cargo test`): `vmx` (field encodings, control reconciliation,
exit/event/segment/region/EPT formats, support), `ept` (tree construction),
`vmcs` pure logic (`error` flag decoding, `fixed` reconciliation), `vmexits`
(the control-register and EPT-violation exit-qualification decoders).

Proven in VMX operation on an Intel processor (via VirtualBox nested VT-x),
all ten self-test checks passing: `VMXON`, `VMCLEAR`/`VMPTRLD`, `VMREAD`/
`VMWRITE` at every field width, `VMLAUNCH`, `VMRESUME` (resume across multiple
exits), guest GPR save/restore, RIP advance past an exiting instruction, a guest
running behind an EPT second translation, and switching between two VMCS.

Written, host-tested where pure, and awaiting the next hardware run: the
`vmexits` dispatch loop and the two self-test checks that drive a guest through
it — one guest that runs `CPUID` and one that runs `RDMSR`, each checked against
the machine's own answer, which together prove the loop emulates an exit and
resumes the guest rather than only stepping over the instruction. The executable
dispatch itself (the `CPUID`/MSR forwarding and RIP advance) runs only in VMX
operation, so it is unverified until those checks pass on Intel.

## Running the self-test on Intel hardware

1. Build with the on-screen log and the self-test:
   `cargo build -p hv-loader -p hv-core --features hv-loader/efifb,hv-core/efifb,hv-core/vmx-selftest`
2. Stage a FAT EFI system partition: `EFI/BOOT/BOOTX64.EFI` ← `hv-loader.efi`,
   `\pulzar.efi` ← `hv-core.efi`, and any valid UEFI PE at
   `\EFI\Microsoft\Boot\bootmgfw.efi` (e.g. a UEFI shell) so the loader's
   guest-manager preload succeeds.
3. For VirtualBox, convert the FAT image to VDI (`qemu-img convert -O vdi`) and
   patch 16 random bytes at offset `0x188` so each image has a unique UUID
   (VirtualBox refuses two disks with the same embedded UUID). VM settings:
   64-bit, EFI on, Secure Boot off, Nested VT-x/AMD-V on.
4. For QEMU/KVM: `-enable-kvm -cpu host` boots the raw image with OVMF directly.

The self-test runs early in `hv-core::bring_up` (right after `start_clock`) and
then halts, because everything below it is SVM and cannot run on Intel. It is a
genuine capability probe, not a mock.

## What remains, roughly in order

1. **VMX exit-dispatch loop** (the `exits` counterpart). The skeleton is the
   `vmexits` crate: `dispatch()` answers `CPUID`, `RDMSR`/`WRMSR`, `HLT`,
   `VMCALL` and a triple fault, and decodes the control-register and
   EPT-violation exit qualifications. What it still lacks, roughly in order: the
   concealment policy on `CPUID` (hide the virtualization extension and the
   hypervisor leaves, as `exits::cpuid` does), an MSR bitmap so only the
   registers that need answering exit (today none exits, since no bitmap is
   programmed — every `RDMSR`/`WRMSR` exits and is forwarded), the handler
   bodies behind the two decoders (which need real guest memory and
   control-register virtualization, items 2 below), I/O, and the remaining exit
   reasons. A refused MSR and both decoded-but-unhandled exits currently stop
   the guest, because giving it the fault it is owed needs item 3.
2. **Real guest memory.** Map actual guest RAM through EPT rather than identity,
   and give the guest its own `CR3` and page tables (or run an unrestricted
   real-mode guest). A partition-equivalent that owns the guest's EPT and memory.
3. **Event injection, correctly.** An injected event is delivered through the
   guest's IDT, not intercepted by the exception bitmap, so a real injection
   path needs the guest to own an IDT. The field format is in `vmx::event`.
4. **APICv** (the `svm::avic` / `vlapic` AVIC counterpart): posted interrupts,
   the virtual-APIC page, the APIC-access page.
5. **Vendor selection in `hv-core`.** Detect AMD (`CPUID 0x8000000A`) versus
   Intel (`CPUID.1:ECX.VMX`) at boot and drive the SVM (`svm`/`vcpu`/`npt`/
   `exits`) or VMX (`vmx`/`vmcs`/`ept`/`vmexits`) backend behind a shared
   interface. Today
   `bring_up` is hardcoded to SVM; this is the refactor that makes one image run
   on either, and the largest remaining piece.

## Gotchas learned the hard way

- `CPUID` and several other instructions cause a VM exit **before** executing, so
  the guest's registers do not reflect their result — the hypervisor emulates
  them and advances RIP.
- Injected exceptions vector through the guest IDT, **not** the exception bitmap.
- The world switch (`vmcs::run`) pins the guest-context pointer to `RCX` and
  recovers it from the stack after exit; `RBP` and the other callee-saved
  registers are preserved by hand; `HOST_RSP`/`HOST_RIP` are written per entry to
  a label inside the asm block.
- Every control word must pass `vmx::Capability::reconcile` against its
  capability MSR, and `CR0`/`CR4` must pass `vmcs::fixed::reconcile` against the
  fixed-bit MSRs, or VM entry fails with a `VM-instruction-error`.
- The self-test guest runs in the host's own address space (`GUEST_CR3` = host
  `CR3`); the EPT check adds an identity EPT on top. A real guest needs its own
  address space.

## Rules that still apply

The whole of `AGENTS.md` binds. In particular: keep the zero-warning lint, doc
and no-stub gates; host-test pure logic; label executable code that has not run
on hardware as unverified; and never claim a hardware result you have not
observed.
