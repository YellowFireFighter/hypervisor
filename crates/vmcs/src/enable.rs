//! Putting one processor into VMX operation.
//!
//! The counterpart to `vcpu`'s `Host::install` on the AMD side, and it has the
//! same shape: check the extension is usable, satisfy the control-register
//! requirements, give the processor the page it needs, and turn it on. The
//! differences are Intel's — a feature-control register that may forbid
//! `VMXON`, control-register bits the processor fixes, and a VMXON region
//! stamped with the VMCS revision — and they all happen here, in the order the
//! architecture requires.
//!
//! This sequence has been exercised in VMX operation on an Intel processor:
//! `VMXON` succeeds and a VMCS can then be made current. The decisions it is
//! built from — [`vmx::support`], [`crate::fixed`] — are also tested where they
//! are defined.

use core::marker::PhantomData;

use vmx::{
    VmxBasic,
    basic::IA32_VMX_BASIC,
    support::{self, FeatureControl, IA32_FEATURE_CONTROL},
};
use x86_64::{
    PhysAddr,
    registers::{
        control::{Cr0, Cr4, Cr4Flags},
        model_specific::Msr,
    },
};

use crate::{
    error::EnterError,
    fixed::{
        self, IA32_VMX_CR0_FIXED0, IA32_VMX_CR0_FIXED1, IA32_VMX_CR4_FIXED0, IA32_VMX_CR4_FIXED1,
    },
    instr,
    msr::rdmsr,
};

/// Proof that the calling processor is in VMX operation.
///
/// Held by the processor that entered it and no other — it carries a
/// `PhantomData<*mut ()>` so it is neither [`Send`] nor [`Sync`], the way
/// `vcpu`'s per-processor state is, because leaving VMX operation or loading a
/// VMCS is only meaningful on the processor that entered it.
#[derive(Debug)]
pub struct Vmx {
    region: PhysAddr,
    basic: VmxBasic,
    _not_send: PhantomData<*mut ()>,
}

impl Vmx {
    /// `IA32_VMX_BASIC` as read while entering, which carries the revision a
    /// VMCS on this processor must be stamped with and the region size.
    #[must_use]
    pub const fn basic(&self) -> VmxBasic {
        self.basic
    }

    /// The physical address of this processor's VMXON region.
    #[must_use]
    pub const fn region(&self) -> PhysAddr {
        self.region
    }
}

/// Puts the calling processor into VMX operation, using the VMXON region whose
/// first doubleword is reachable at `region_header` and whose physical address
/// is `region_phys`.
///
/// On success the processor is in VMX operation and the returned [`Vmx`] is the
/// token for driving a VMCS on it.
///
/// # Errors
///
/// [`EnterError::Unsupported`] if `CPUID` does not report VMX,
/// [`EnterError::DisabledByFirmware`] if a locked feature-control register
/// forbids it, or [`EnterError::Vmxon`] if the instruction itself fails.
///
/// # Safety
///
/// `region_header` must point at the first doubleword of a page that is
/// `region_phys` in physical memory, mapped writable, sized as a VMXON region
/// and used by nothing else for as long as VMX operation lasts. This must be
/// called at most once per processor without an intervening leave, and the
/// processor must be running in long mode with paging on, which `hv-core`
/// always is by the time this runs.
pub unsafe fn enter(region_header: *mut u32, region_phys: PhysAddr) -> Result<Vmx, EnterError> {
    // Leaf 1 is implemented on every x86-64 processor, so this reads a real
    // feature word rather than whatever a missing leaf would return.
    let ecx = core::arch::x86_64::__cpuid(support::FEATURE_LEAF).ecx;
    if !support::supported(ecx) {
        return Err(EnterError::Unsupported);
    }

    // SAFETY: the feature-control register exists on every VMX-capable
    // processor, which the check above established this is.
    let feature_control = FeatureControl::from_bits_retain(unsafe { rdmsr(IA32_FEATURE_CONTROL) });
    if feature_control.contains(FeatureControl::LOCK) {
        if !feature_control.vmxon_allowed_outside_smx() {
            return Err(EnterError::DisabledByFirmware);
        }
    } else {
        // Unlocked: configure it ourselves and lock it, which is what firmware
        // would otherwise have done.
        let configured = feature_control | FeatureControl::LOCK | FeatureControl::VMXON_OUTSIDE_SMX;
        // SAFETY: the register is unlocked, so it is writable; the value only
        // permits VMXON outside SMX and sets the lock.
        unsafe { Msr::new(IA32_FEATURE_CONTROL).write(configured.bits()) };
    }

    // SAFETY: IA32_VMX_BASIC exists on a VMX-capable processor.
    let basic = VmxBasic::from_bits(unsafe { rdmsr(IA32_VMX_BASIC) });

    // Bring CR0 and CR4 within the bits VMX fixes, and set CR4.VMXE, before
    // VMXON — the instruction faults otherwise.
    // SAFETY: the four fixed-bit registers exist on a VMX-capable processor.
    let (cr0_fixed0, cr0_fixed1, cr4_fixed0, cr4_fixed1) = unsafe {
        (
            rdmsr(IA32_VMX_CR0_FIXED0),
            rdmsr(IA32_VMX_CR0_FIXED1),
            rdmsr(IA32_VMX_CR4_FIXED0),
            rdmsr(IA32_VMX_CR4_FIXED1),
        )
    };
    let cr0 = fixed::reconcile(Cr0::read_raw(), cr0_fixed0, cr0_fixed1);
    let cr4 = fixed::reconcile(
        Cr4::read_raw() | Cr4Flags::VIRTUAL_MACHINE_EXTENSIONS.bits(),
        cr4_fixed0,
        cr4_fixed1,
    );
    // SAFETY: both values were reconciled against the processor's own fixed-bit
    // registers, so they are values VMX accepts; this runs in long mode where
    // clearing paging or protection is not among the bits reconcile can clear
    // (those are forced on by CR0_FIXED0).
    unsafe {
        Cr0::write_raw(cr0);
        Cr4::write_raw(cr4);
    }

    // Stamp the revision into the region's first doubleword.
    // SAFETY: the caller guarantees `region_header` points at the writable
    // first doubleword of the VMXON region.
    unsafe { region_header.write(vmx::region::vmxon_header(basic.revision())) };

    // SAFETY: VMX is supported, CR4.VMXE is set, feature control permits VMXON,
    // and the region is stamped and exclusively owned — the preconditions of
    // the instruction.
    unsafe { instr::vmxon(region_phys) }
        .ok()
        .map_err(EnterError::Vmxon)?;

    Ok(Vmx {
        region: region_phys,
        basic,
        _not_send: PhantomData,
    })
}
