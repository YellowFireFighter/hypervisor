# Intel VMX port — status and roadmap

citrine is an AMD SVM hypervisor. An Intel VMX backend is being built alongside
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
| `vmexits` | Exit dispatch and the run loop: `dispatch()` answers each exit and returns `Flow` (resume / vmcall / stop), with `CPUID` concealment, `RDMSR`/`WRMSR` forwarding, `HLT`, `VMCALL`, triple fault, and the control-register and EPT-violation handlers. `run()` drives a guest to a stop or a hypercall, resolving EPT violations through a `Partition`. | `exits` |
| `crates/hv-core/src/selftest.rs` | Feature-gated (`vmx-selftest`) battery that drives all of the above on real hardware and reports `[PASS]`/`[FAIL]` per check. | — |

## Verification status

Host unit tests (`cargo test`): `vmx` (field encodings, control reconciliation,
exit/event/segment/region/EPT formats, support), `ept` (tree construction),
`vmcs` pure logic (`error` flag decoding, `fixed` reconciliation), `vmexits`
(the control-register and EPT-violation exit-qualification decoders).

Proven in VMX operation on an Intel processor (via VirtualBox nested VT-x), the
self-test battery passing in full. The checks, by area:

- **Enable and VMCS management:** `VMXON`, `CR4.VMXE`, `VMCLEAR`/`VMPTRLD`,
  `VMREAD`/`VMWRITE` at every field width, switching between two VMCS.
- **World switch:** `VMLAUNCH`, `VMRESUME` across multiple exits, guest GPR
  save/restore, RIP advance past an exiting instruction.
- **Second translation:** a guest behind an identity EPT; a guest in its own
  address space (own `CR3` and page tables) behind a non-identity EPT; an EPT
  violation demand-mapped into the live tree and flushed with `INVEPT`, the
  faulting access restarted.
- **Dispatch loop:** a guest driven through `vmexits::dispatch()`, its `CPUID`
  and `RDMSR`/`WRMSR` exits emulated and checked against the machine's own
  answer; `CPUID` concealing the virtualization extension and the hypervisor
  leaves; `IA32_FEATURE_CONTROL` answered as firmware-locked; a control-register
  access decoded from its qualification.
- **Event injection:** an exception, and one carrying an error code, injected
  through a guest's own IDT and delivered to its handler.

Written, and awaiting the next hardware run: the first APICv slice — the TPR
shadow virtualizing `CR8` to the virtual-APIC page without an exit, and the TPR
threshold taking the exit when the shadow drops below it — each with a self-test
check, unverified until it runs on Intel.

The firmware-guest entry probe (the `vmx-boot` feature, in `hv-core`'s
`vmxboot` module) has had one hardware run. On the Intel branch of bring-up it
enters VMX operation, builds an identity EPT over all of physical memory,
programs a VMCS from the captured firmware save area — reconciling `CR0`/`CR4`
against the fixed bits and converting each segment's packed attributes to the
VMCS access-rights word — and enters that state as a guest at a one-instruction
stub that calls straight back into the host. It is the first piece of the guest
path proper rather than a mechanism check: firmware's captured `RIP` is zero
(left for the entry to fill), so there is no portal yet and the stub stands in
for one.

On the first run (an AMI firmware, 19 GiB of RAM) VMX operation, the EPT build
and the VMCS programming all succeeded, and the entry failed with basic exit
reason 33, *invalid guest state*. That exit names no specific check, so the
probe now restates the checks itself (`vmx::check`, fed by
`vmcs::inspect::guest_state`), logs every rule the programmed state breaks
before trying the entry, and dumps the guest state raw if the entry still
fails. Two adjustments were made for the most likely causes, unverified until
the next run: a null firmware task register becomes a minimal busy 64-bit one
(VMX requires a usable TSS; AMD's world switch does not check), and every
loaded code or data segment is marked accessed.

## Running the self-test on Intel hardware

1. Build with the on-screen log and the self-test:
   `cargo build -p hv-loader -p hv-core --features hv-loader/efifb,hv-core/efifb,hv-core/vmx-selftest`
2. Stage a FAT EFI system partition: `EFI/BOOT/BOOTX64.EFI` ← `hv-loader.efi`,
   `\citrine.efi` ← `hv-core.efi`, and any valid UEFI PE at
   `\EFI\Microsoft\Boot\bootmgfw.efi` (e.g. a UEFI shell) so the loader's
   guest-manager preload succeeds.
3. For VirtualBox, convert the FAT image to VDI (`qemu-img convert -O vdi`) and
   patch 16 random bytes at offset `0x188` so each image has a unique UUID
   (VirtualBox refuses two disks with the same embedded UUID). VM settings:
   64-bit, EFI on, Secure Boot off, Nested VT-x/AMD-V on.
4. For QEMU/KVM: `-enable-kvm -cpu host` boots the raw image with OVMF directly.

The self-test runs early in `hv-core::bring_up`, before the ACPI and clock
bring-up, and then halts, because everything below it is SVM and cannot run on
Intel. It is a genuine capability probe, not a mock.

To run the firmware-guest entry probe instead, swap the feature:

```sh
cargo build -p hv-loader -p hv-core \
  --features hv-loader/efifb,hv-core/efifb,hv-core/vmx-boot,hv-loader/no-guest
```

and stage the media the same way. `no-guest` is used because the probe halts
before any guest boot manager runs, so the loader needs none and must not refuse
to boot on a machine with several ESPs or none. The probe runs at the vendor
branch, after the clock bring-up, and halts with its report afterward.

## What is done, and what remains

Done, and proven on Intel (see above): the **exit-dispatch loop** (`vmexits`) —
`CPUID` with concealment, `RDMSR`/`WRMSR` with concealment, `HLT`, `VMCALL`,
triple fault, and the control-register and EPT-violation handlers behind their
decoders; **real guest memory** — a guest with its own `CR3` and page tables
behind a non-identity EPT, with demand mapping on an EPT violation; and **event
injection** through a guest's own IDT, with and without an error code.

Remaining, roughly in order:

1. **Finish the dispatch loop.** An MSR bitmap, so only the registers that need
   answering exit (today no bitmap is programmed, so every `RDMSR`/`WRMSR`
   exits and is forwarded); I/O; and the remaining exit reasons. A refused MSR
   still stops the guest, because giving it the fault it is owed is part of the
   injection work the self-test now proves but the dispatch does not yet apply.
2. **APICv** (the `svm::avic` / `vlapic` AVIC counterpart). The TPR shadow is
   the first slice; what remains is the APIC-access page, APIC-register
   virtualization, virtual-interrupt delivery, and posted interrupts.
3. **Vendor selection in `hv-core`.** The detection and the selection seam are
   in place: `processor::vendor()` reads the vendor string and `bring_up`
   branches on it, so the image chooses the SVM (`svm`/`vcpu`/`npt`/`exits`) or
   VMX (`vmx`/`vmcs`/`ept`/`vmexits`) backend rather than assuming SVM. Today the
   AMD branch is the whole guest path, and the Intel branch either stops with a
   report (`CoreError::IntelBackendNotWired`) or, under the `vmx-boot` feature,
   runs the firmware-guest entry probe above and then halts. The run loop driving
   `vmexits::dispatch` now exists as `vmexits::run`, taking a `Partition` for the
   guest's memory, and the probe is the first caller to enter a VMCS programmed
   from real firmware state. What remains is the rest of that path: a
   partition-equivalent that owns the firmware guest's full EPT and memory, the
   portal (so the guest resumes into the boot manager rather than a stub), SMP
   bring-up, and device interposition — the VMX counterparts of what
   `partition`/`portal` give the SVM side. Unlike the mechanisms above, finishing
   it is validated by booting a real guest on an Intel machine, not by the
   self-test battery.

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
- Most self-test guests run in the host's own address space (`GUEST_CR3` = host
  `CR3`), which keeps the probe for one mechanism from depending on the paging
  of another; the own-address-space and demand-paging guests instead build their
  own `CR3` and page tables and run behind a non-identity EPT, overriding
  `GUEST_CR3` after `program_guest`.
- A guest page added to a live EPT is not visible until `INVEPT` drops the stale
  translation; an EPT violation does not advance the guest, so the faulting
  instruction restarts on re-entry.
- Under the TPR shadow, `CR8` and the task-priority register go to the
  virtual-APIC page, not the real controller; VM entry itself exits if the
  seeded `VTPR[7:4]` is below the TPR threshold, so the threshold probe seeds it
  above and lets the guest's write cross below.

## Rules that still apply

The whole of `AGENTS.md` binds. In particular: keep the zero-warning lint, doc
and no-stub gates; host-test pure logic; label executable code that has not run
on hardware as unverified; and never claim a hardware result you have not
observed.
