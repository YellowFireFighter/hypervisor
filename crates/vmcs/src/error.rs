//! What a VMX instruction reports through the flags, and the errors that can
//! stop VMX operation being entered.
//!
//! Every VMX instruction tells software how it went not through a return value
//! but through the arithmetic flags, in one of three ways the architecture
//! fixes: it succeeded; it failed with no current VMCS to record an error in
//! (`VMfailInvalid`, carry set); or it failed with a current VMCS, which holds
//! a numeric reason in its VM-instruction-error field (`VMfailValid`, zero
//! set). [`Outcome::from_flags`] turns a captured `RFLAGS` into which of the
//! three happened, and it is pure — the one part of this crate's instruction
//! handling that runs under test on the host.

use thiserror::Error;

/// Carry flag: `VMfailInvalid`.
const CARRY: u64 = 1 << 0;
/// Zero flag: `VMfailValid`.
const ZERO: u64 = 1 << 6;

/// How a VMX instruction reported it went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The instruction succeeded.
    Success,
    /// The instruction failed and there was no current VMCS to record why in.
    /// This is what a `VMREAD` or `VMWRITE` with nothing loaded reports, and a
    /// `VMPTRLD` of a region whose revision does not match the processor.
    FailInvalid,
    /// The instruction failed with a current VMCS, whose VM-instruction-error
    /// field now holds the numeric reason.
    FailValid,
}

impl Outcome {
    /// Which outcome the flags left after a VMX instruction describe.
    ///
    /// Carry takes precedence over zero: the architecture sets exactly one of
    /// them on a failure and neither on success, so testing carry first and
    /// zero second names all three cases without ambiguity.
    #[must_use]
    pub const fn from_flags(rflags: u64) -> Self {
        if rflags & CARRY != 0 {
            Self::FailInvalid
        } else if rflags & ZERO != 0 {
            Self::FailValid
        } else {
            Self::Success
        }
    }

    /// `Ok` on success, or the matching [`VmFail`] otherwise.
    ///
    /// # Errors
    ///
    /// [`VmFail::Invalid`] for `VMfailInvalid` and [`VmFail::Valid`] for
    /// `VMfailValid`; the caller reads the VM-instruction-error field for the
    /// reason behind the latter.
    pub const fn ok(self) -> Result<(), VmFail> {
        match self {
            Self::Success => Ok(()),
            Self::FailInvalid => Err(VmFail::Invalid),
            Self::FailValid => Err(VmFail::Valid),
        }
    }
}

/// A VMX instruction that did not succeed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum VmFail {
    /// `VMfailInvalid`: the instruction failed with no current VMCS.
    #[error("a VMX instruction failed with no current VMCS")]
    Invalid,
    /// `VMfailValid`: the instruction failed; the VM-instruction-error field
    /// holds the reason.
    #[error("a VMX instruction failed; see the VM-instruction-error field")]
    Valid,
}

/// Why this processor could not be put into VMX operation.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum EnterError {
    /// `CPUID` does not report VMX on this processor.
    #[error("this processor does not support VMX")]
    Unsupported,
    /// The feature-control register is locked in a state that forbids `VMXON`
    /// outside SMX, so firmware disabled VMX and software cannot override it.
    #[error("VMX is disabled by a locked IA32_FEATURE_CONTROL")]
    DisabledByFirmware,
    /// `VMXON` itself failed.
    #[error("VMXON failed: {0}")]
    Vmxon(VmFail),
}

#[cfg(test)]
mod tests {
    use super::{Outcome, VmFail};

    #[test]
    fn clear_flags_are_success() {
        assert_eq!(Outcome::from_flags(0), Outcome::Success);
        assert_eq!(Outcome::from_flags(0).ok(), Ok(()));
    }

    #[test]
    fn carry_is_fail_invalid_and_wins_over_zero() {
        assert_eq!(Outcome::from_flags(1 << 0), Outcome::FailInvalid);
        // The architecture never sets both, but carry is tested first.
        assert_eq!(
            Outcome::from_flags((1 << 0) | (1 << 6)),
            Outcome::FailInvalid
        );
        assert_eq!(Outcome::from_flags(1 << 0).ok(), Err(VmFail::Invalid));
    }

    #[test]
    fn zero_alone_is_fail_valid() {
        assert_eq!(Outcome::from_flags(1 << 6), Outcome::FailValid);
        assert_eq!(Outcome::from_flags(1 << 6).ok(), Err(VmFail::Valid));
    }

    #[test]
    fn unrelated_flags_do_not_change_the_verdict() {
        // Sign, parity and adjust set, but neither carry nor zero.
        let rflags = (1 << 7) | (1 << 2) | (1 << 4);
        assert_eq!(Outcome::from_flags(rflags), Outcome::Success);
    }
}
