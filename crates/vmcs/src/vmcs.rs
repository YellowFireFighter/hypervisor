//! A VMCS made current on this processor, and typed access to its fields.
//!
//! Once [`Vmx`](crate::Vmx) has put the processor into VMX operation, a guest
//! is described by a VMCS: a page stamped with the revision, cleared, and made
//! current, after which every [`read`](Vmcs::read) and [`write`](Vmcs::write)
//! goes to it. This is the Intel counterpart to allocating and filling a VMCB,
//! except that a VMCB is filled by storing into its fields and a VMCS is filled
//! one `VMWRITE` at a time.
//!
//! Programming the control words goes through [`reconcile_control`], which is
//! where the capability registers from [`vmx::Capability`] meet the field:
//! a control word is never written raw, only after being brought within what
//! the processor allows.
//!
//! The field encodings and the reconciliation are checked where they are
//! defined, and the VMCS lifecycle here — `VMCLEAR`, `VMPTRLD`, then a
//! `VMWRITE`/`VMREAD` round-trip of a field — has been exercised in VMX
//! operation on an Intel processor.

use vmx::{Capability, Field, FieldEncoding, VmxBasic};
use x86_64::PhysAddr;

use crate::{error::VmFail, instr};

/// A VMCS that is current on this processor.
///
/// Holding one is the claim that the VMCS at [`region`](Vmcs::region) was made
/// current with `VMPTRLD` and has not been displaced since, so the field
/// accessors reach it. It is not [`Send`]: a VMCS is current on one processor,
/// and `instr`'s wrappers act on whatever is current on the processor that runs
/// them.
#[derive(Debug)]
pub struct Vmcs {
    region: PhysAddr,
    launched: bool,
}

impl Vmcs {
    /// Stamps the revision into the VMCS at `region_header`, clears it, and
    /// makes it current.
    ///
    /// # Errors
    ///
    /// The [`VmFail`] from `VMCLEAR` or `VMPTRLD` if either is rejected — a
    /// region of the wrong size, or a revision that does not match the
    /// processor.
    ///
    /// # Safety
    ///
    /// The calling processor must be in VMX operation. `region_header` must
    /// point at the writable first doubleword of a page that is `region` in
    /// physical memory, sized as a VMCS and owned by nothing else, and `basic`
    /// must be this processor's [`VmxBasic`] so the revision matches.
    pub unsafe fn activate(
        region_header: *mut u32,
        region: PhysAddr,
        basic: VmxBasic,
    ) -> Result<Self, VmFail> {
        // SAFETY: the caller guarantees `region_header` is the writable first
        // doubleword of the VMCS region.
        unsafe { region_header.write(vmx::region::vmcs_header(basic.revision())) };
        // SAFETY: the caller guarantees VMX operation and a valid region; a
        // fresh VMCS is cleared before it is loaded.
        unsafe { instr::vmclear(region) }.ok()?;
        // SAFETY: the region was just cleared with a matching revision.
        unsafe { instr::vmptrld(region) }.ok()?;
        Ok(Self {
            region,
            launched: false,
        })
    }

    /// The physical address of this VMCS.
    #[must_use]
    pub const fn region(&self) -> PhysAddr {
        self.region
    }

    /// Whether this VMCS has been launched, so the run loop resumes rather than
    /// launches it.
    #[must_use]
    pub const fn launched(&self) -> bool {
        self.launched
    }

    /// Records that this VMCS has been launched.
    pub fn mark_launched(&mut self) {
        self.launched = true;
    }

    /// Reads `field` from this VMCS.
    ///
    /// # Errors
    ///
    /// [`VmFail`] if the field is not one the processor's VMCS has, or this
    /// VMCS is no longer current.
    ///
    /// # Safety
    ///
    /// This VMCS must still be the current one on this processor.
    pub unsafe fn read(&self, field: FieldEncoding) -> Result<u64, VmFail> {
        // SAFETY: the caller guarantees this VMCS is still current.
        unsafe { instr::vmread(field) }
    }

    /// Writes `value` into `field` of this VMCS.
    ///
    /// # Errors
    ///
    /// [`VmFail`] if the field is read-only or absent, or this VMCS is no
    /// longer current.
    ///
    /// # Safety
    ///
    /// This VMCS must still be the current one on this processor.
    pub unsafe fn write(&self, field: FieldEncoding, value: u64) -> Result<(), VmFail> {
        // SAFETY: the caller guarantees this VMCS is still current.
        unsafe { instr::vmwrite(field, value) }
    }

    /// Advances the guest's `RIP` past the instruction it exited on, which is
    /// what resuming after an instruction the host emulated requires.
    ///
    /// The processor records how long that instruction was; this reads it and
    /// the current `RIP` and writes their sum back, so the next entry resumes
    /// at the following instruction rather than re-executing the one that
    /// exited.
    ///
    /// # Errors
    ///
    /// [`VmFail`] if either field cannot be read or written, or this VMCS is no
    /// longer current.
    ///
    /// # Safety
    ///
    /// This VMCS must still be the current one on this processor, and the last
    /// exit must have been on an instruction whose length the processor
    /// recorded — which an exit caused by executing an instruction always does.
    pub unsafe fn advance_past_instruction(&self) -> Result<(), VmFail> {
        // SAFETY: the caller guarantees this VMCS is still current.
        unsafe {
            let rip = self.read(Field::GUEST_RIP)?;
            let length = self.read(Field::VM_EXIT_INSTRUCTION_LENGTH)?;
            self.write(Field::GUEST_RIP, rip.wrapping_add(length))
        }
    }

    /// Writes a control word into `field` after reconciling `desired` against
    /// `capability`, so the value is one the processor will accept.
    ///
    /// This is the only way a control word should reach a VMCS: a bit outside
    /// what the capability register allows fails VM entry, and
    /// [`Capability::reconcile`] is what removes it.
    ///
    /// # Errors
    ///
    /// [`VmFail`] if the write is rejected or this VMCS is no longer current.
    ///
    /// # Safety
    ///
    /// This VMCS must still be the current one on this processor.
    pub unsafe fn reconcile_control(
        &self,
        field: FieldEncoding,
        desired: u32,
        capability: Capability,
    ) -> Result<(), VmFail> {
        // SAFETY: the caller guarantees this VMCS is still current.
        unsafe { self.write(field, u64::from(capability.reconcile(desired))) }
    }
}
