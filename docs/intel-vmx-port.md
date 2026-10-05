# Intel VMX port — status and roadmap

citrine is an AMD SVM hypervisor. An Intel VMX backend is being built alongside
the SVM one so a single image can eventually run on either vendor. The
foundational layers exist and are proven on Intel hardware; the higher layers
that a real OS guest needs do not yet exist. This file is the map of what is
done, how it is tested, and what remains.

## Handoff — current state and what to do next (read this first)

**The goal.** Boot a real Intel Windows PC behind citrine from a USB stick:
citrine loads from the stick, captures firmware, re-enters firmware as a VMX
guest through the "portal", lets it start Windows' boot manager, and keeps the
machine virtualized from then on. The AMD/SVM side of this already works; the
Intel/VMX side is being brought up to match it.

**Where the work is.** The active path is the firmware-guest VMX probe behind
the `vmx-boot` feature: `crates/hv-core/src/vmxboot.rs` (the driver) plus
`crates/vmcs/src/controls.rs` (VMCS control programming), `crates/ept`,
`crates/vmexits`, `crates/portal`. On an Intel machine `hv-core` takes the
Intel branch in `crates/hv-core/src/main.rs` (~line 240) and calls
`vmxboot::attempt`.

**What already works on the real hardware** (confirmed by the user's boot
photos): VM entry with real firmware state passes every consistency check; the
guest runs behind an identity EPT with firmware's own paging; the portal runs;
the local APIC is virtualized so firmware's INIT/SIPI no longer reset the box.

**The blocker being chased.** After APIC virtualization the firmware guest no
longer resets but *stalls*: it spins in a tight poll loop (seen bouncing between
two addresses, e.g. `0x65498041` ↔ `0x65492492`) and never progresses, because
its periodic timer/event interrupt was not reaching it. Fixes tried, newest
last:
1. Fake-tick the virtual-APIC timer and inject its vector — didn't help (the
   firmware timer's LVT vector is below the APIC's legal range; removed).
2. Forward the guest's end-of-interrupt to the real local APIC
   (`ApicForward`) — correct but not sufficient alone.
3. **Interrupt reflection (current, newest commit).** External-interrupt
   exiting + acknowledge-on-exit: every physical interrupt now VM-exits with its
   vector, and the host injects it into the guest's own IDT (holding it behind an
   interrupt-window when the guest has interrupts masked). This is the VMX
   counterpart of what the AMD world switch does. **Awaiting a hardware test.**

**How to read the next boot photo** (the probe logs to the screen, newest at the
bottom):
- `external interrupts exit to the host and are reflected into the guest` →
  reflection armed.
- Sample lines carry `N interrupts delivered` and `M end-of-interrupts
  forwarded`, plus `guest running at rip 0x…`. **Win:** RIPs move to *new*
  regions and `interrupts delivered` climbs. **Still stuck + `0 delivered`:**
  interrupts still aren't arriving (look elsewhere — PIC/IOAPIC routing, the
  real-APIC dump lines `real APIC isr/irr`). **Stuck + `N delivered`:**
  interrupts flow but firmware waits on something else — decode the loop from the
  `code at 0x…: [bytes]` and `spin regs …` lines the probe prints.

**If reflection works**, the next milestones are: firmware reaches
`ExitBootServices` (Windows' boot manager ran under citrine — the big proof),
then the unbuilt post-`ExitBootServices` VMX path: starting the other CPU cores
(SMP), concealing the portal, and interposing on devices — the VMX counterparts
of what `partition`/`portal`/`vlapic` give the SVM side.

**The AMD path is the blueprint.** The working interrupt virtualization lives in
`crates/vlapic` (a full virtual local APIC) + `crates/inject` + `crates/exits`,
driven from `exits::Dispatcher::run`. "Make the Intel side like AMD" ultimately
means either reflection (done, software injection) maturing into a `vlapic`-style
model, or enabling hardware virtual-interrupt-delivery (VID/APICv) — the Intel
equivalent of AVIC. A reference Intel VMX hypervisor the user pointed at:
`github.com/tomtzook/hype` (note: it hyperjacks the running system and does *not*
virtualize the APIC, so interrupts pass straight through — citrine virtualizes
the APIC for reset protection, which is why it must reflect).

**Testing (important).** This repo builds and boots under QEMU, and QEMU's
software (TCG) mode emulates **AMD SVM** — so `-cpu max` runs citrine's *SVM*
path and is good for boot/regression checks. It does **not** emulate **Intel
VMX**, and the cloud dev box has no `/dev/kvm` / nested VT-x, so the VMX
firmware-guest path **can only be validated on real Intel hardware**. Pre-flight
a build with:
```sh
qemu-system-x86_64 -machine q35 -accel tcg -cpu max -m 4G -smp 1 \
  -drive if=pflash,format=raw,readonly=on,file=/usr/share/OVMF/OVMF_CODE_4M.fd \
  -drive if=pflash,format=raw,file=<writable copy of OVMF_VARS_4M.fd> \
  -drive if=none,id=esp,format=raw,file=fat:rw:<esp dir> \
  -device ide-hd,drive=esp,bus=ide.0,bootindex=0 \
  -serial file:boot.log -display none -no-reboot -no-shutdown
```
(build that esp dir's images **without** `efifb` so logs go to serial).

**Build the bootable USB image for the user** (on-screen log via `efifb`, Intel
firmware-guest probe, Windows boot manager as the guest):
```sh
cargo build -p hv-loader -p hv-core \
  --features hv-loader/efifb,hv-core/efifb,hv-core/vmx-boot
# then stage a GPT+FAT32 ESP: BOOTX64.EFI = hv-loader.efi, \citrine.efi =
# hv-core.efi, package with sgdisk/mtools, gzip, and send to the user to flash.
```
Swap `vmx-boot`→`vmx-selftest` (add `hv-loader/no-guest`) to run the self-test
battery instead. Gates before every commit: `cargo fmt --all -- --check`,
`cargo clippy --all-targets`, `cargo clippy -p hv-core --features vmx-boot
--target x86_64-unknown-uefi`, and the same with `vmx-selftest`.

**Honest scope.** Reaching `ExitBootServices` proves Windows' boot manager runs
under citrine; carrying it all the way to a Windows desktop is a large amount of
further work (the whole post-firmware OS path). No single boot gets from here to
the desktop.

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
- **Concealment:** a guest executing a VMX instruction (`VMXOFF`) is refused
  with an invalid-opcode exception delivered through its own IDT, the
  counterpart to the AMD side's refusal of SVM instructions.
- **Event injection:** an exception, and one carrying an error code, injected
  through a guest's own IDT and delivered to its handler.

The first APICv slice has now run on Intel. The TPR shadow (virtualizing `CR8`
to the virtual-APIC page without an exit) and the TPR threshold (taking the exit
when the shadow drops below it) both pass. The fuller APICv checks written
alongside them — the APIC-access page exiting, APIC-register virtualization
reading from the virtual-APIC page, and virtual-interrupt delivery — fail on
hardware: they are set up without a second translation, which real APIC
virtualization needs, and virtual-interrupt delivery needs its own control and
guest-interrupt-status setup besides. They are deferred: a guest runs on the
physical APIC through the identity EPT without any of them, so none is on the
path to booting a guest, and getting them right is its own effort to be done
when an OS needs interrupt virtualization, not before.

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

On the second run both adjustments held: the guest state passed every
restated check and **VM entry with real firmware state succeeded**. The guest
then exited on a read of `CR3` (`MOV RAX, CR3`; CR3-store exiting is forced on
by the non-TRUE capability register) — not an instruction of the stub, but the
sequence firmware's common exception and interrupt entry uses to save the
control registers. So an event was delivered through firmware's IDT before the
stub ran: most likely the firmware timer interrupt left pending while the host
ran with interrupts off, possibly a fault fetching the stub. The probe now
enters the stub with `RFLAGS.IF` clear and intercepts every exception, and
logs the guest's `RIP`/`RSP`/`RFLAGS` on any stop and the vector, error code
and qualification of an intercepted exception; unverified until the next run.
Running firmware itself (rather than the stub) will need its interrupts
delivered and its `CR3` accesses either not exiting (the TRUE controls, where
`IA32_VMX_BASIC` bit 55 offers them) or emulated.

On the third run **the stub reached its `VMCALL`**: VM entry, the world switch
and the identity EPT carry the captured firmware state on real Intel hardware,
and the stub, which lives in the hypervisor's chunk, executes under firmware's
own page tables. Firmware's captured `RFLAGS` was `0x206`, so interrupts were
enabled, which confirms the second run's exit as firmware's pending timer
interrupt being delivered through firmware's IDT. This is the first milestone
of the guest path. The next is resuming firmware itself rather than a stub,
which on this side still lacks: the portal's host calls (the portal is written
with AMD's `VMMCALL`, which faults on Intel), a guest whose interrupts and `CR3`
accesses are handled, and the partition and `ExitBootServices` handoff the SVM
side builds in `partition` and `portal`.

The fourth build takes the next step on this probe: it lets firmware's own code
run as the guest rather than isolating the stub. The stub keeps firmware's
interrupts enabled, so the pending timer interrupt is delivered through
firmware's IDT to firmware's handler, which returns to the stub's `VMCALL`; and
the guest's `CR3` accesses stop exiting (`vmcs::controls::relax_cr3_exiting`,
using the true capability registers) so the handler's control-register saves
do not stop it. Unverified until it runs: the expected outcome is still the
`VMCALL`, now reached only after firmware's handler has executed.

The portal path has since been wired on the Intel side, replacing the stub. The
portal blob emits `vmcall` under the `portal` crate's `vmx` feature; the probe
places the portal, programs the VMCS from firmware state with the portal as the
entry and firmware's stack realigned for the calls it makes, resumes firmware
there behind the identity EPT, and answers the portal's notifications
(`LoaderUnloaded`, `LoaderSkipped`, `ExitSucceeded`, `StartReturned`) as the
guest leaves firmware. Built with `no-guest`, no boot manager is preloaded, so
the portal's `StartImage` is called on the loader's own already-started handle,
which firmware rejects — the probe reports the returned status. Reaching that
report proves the Intel portal ran: it patched firmware's live boot-services
table, recomputed its CRC through firmware's own `CalculateCrc32`, and called
firmware's `StartImage`, all as the VMX guest. Unverified until it runs on
hardware.

What remains to boot a real guest: building without `no-guest` so the loader
preloads the operating system's boot manager for `StartImage` to launch, and
the VMX path past `ExitBootServices` — a partition owning the guest's full EPT
and memory, SMP bring-up, portal concealment, and device interposition — the
counterparts of what `partition`/`portal`/`exits` give the SVM side.

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
