//! Decoding a control-register access from its exit qualification.
//!
//! When a guest's access to a control register exits, the processor describes
//! it in the exit qualification rather than leaving the instruction to be
//! decoded: which register, whether the access moved a value in or out, and
//! which general register the value came from or went to. Reading those fields
//! out is pure, which is why it is stated and tested here rather than at the
//! exit site.
//!
//! No control-register virtualization is built yet, so the decode exists to
//! report an access precisely rather than to act on one; a guest that touches a
//! control register stops with the access named.

/// Which direction a control-register access moved a value, from bits 5:4 of
/// the exit qualification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `MOV` to a control register: a general register's value was written in.
    ToRegister,
    /// `MOV` from a control register: the register was read into a general
    /// register.
    FromRegister,
    /// `CLTS`: the task-switched bit of `CR0` was cleared.
    ClearTaskSwitched,
    /// `LMSW`: the low word of `CR0` was loaded.
    LoadMachineStatus,
}

impl Kind {
    /// The kind a two-bit access-type code names.
    #[must_use]
    const fn from_code(code: u8) -> Self {
        match code {
            0 => Self::ToRegister,
            1 => Self::FromRegister,
            2 => Self::ClearTaskSwitched,
            // The field is two bits, so three is the only remaining value.
            _ => Self::LoadMachineStatus,
        }
    }
}

/// A control-register access, as the exit qualification describes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    /// The control register's number: 0, 3, 4 or 8.
    pub register: u8,
    /// Which direction the access moved a value.
    pub kind: Kind,
    /// The general register the value came from or went to, for a `MOV`. It is
    /// the architectural register number, zero for `RAX` through fifteen for
    /// `R15`, and is meaningful only when [`kind`](Self::kind) is a `MOV`.
    pub gpr: u8,
}

/// The control-register number occupies the low four bits.
const REGISTER_MASK: u64 = 0xF;
/// The access type is in bits 5:4.
const KIND_SHIFT: u64 = 4;
/// The access type is two bits wide.
const KIND_MASK: u64 = 0b11;
/// The general-register number is in bits 11:8.
const GPR_SHIFT: u64 = 8;
/// The general-register number is four bits wide.
const GPR_MASK: u64 = 0xF;

impl Access {
    /// Takes a control-register exit qualification apart.
    #[must_use]
    pub const fn decode(qualification: u64) -> Self {
        Self {
            register: (qualification & REGISTER_MASK) as u8,
            kind: Kind::from_code(((qualification >> KIND_SHIFT) & KIND_MASK) as u8),
            gpr: ((qualification >> GPR_SHIFT) & GPR_MASK) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Access, Kind};

    #[test]
    fn a_move_to_cr0_from_rax_decodes() {
        // Register 0, access type 0 (MOV to CR), general register 0 (RAX).
        let access = Access::decode(0);
        assert_eq!(access.register, 0);
        assert_eq!(access.kind, Kind::ToRegister);
        assert_eq!(access.gpr, 0);
    }

    #[test]
    fn a_move_from_cr4_into_r9_decodes() {
        // Register 4, access type 1 (MOV from CR) in bits 5:4, register 9 in
        // bits 11:8.
        let qualification = 4 | (1 << 4) | (9 << 8);
        let access = Access::decode(qualification);
        assert_eq!(access.register, 4);
        assert_eq!(access.kind, Kind::FromRegister);
        assert_eq!(access.gpr, 9);
    }

    #[test]
    fn clts_and_lmsw_decode_from_their_codes() {
        assert_eq!(Access::decode(2 << 4).kind, Kind::ClearTaskSwitched);
        assert_eq!(Access::decode(3 << 4).kind, Kind::LoadMachineStatus);
    }

    #[test]
    fn fields_above_the_decoded_ones_are_ignored() {
        // The LMSW source data sits in bits 31:16 and must not bleed into the
        // decoded fields.
        let qualification = 8 | (1 << 4) | (0xF << 8) | (0xDEAD << 16);
        let access = Access::decode(qualification);
        assert_eq!(access.register, 8);
        assert_eq!(access.kind, Kind::FromRegister);
        assert_eq!(access.gpr, 0xF);
    }
}
