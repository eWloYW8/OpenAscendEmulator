use crate::isa::c220::load_va::C220LoadVaInstruction;
use crate::memory::ub::UbMemory;
use crate::sim::c220::vector::{C220VectorError, C220VectorReadAccess};

const LOAD_BYTES: usize = 32;
const HALF_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadVaIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220LoadVaInstruction,
    pub source_address: u64,
}

impl C220LoadVaIssue {
    pub const fn read_access(&self) -> C220VectorReadAccess {
        C220VectorReadAccess {
            source_index: 0,
            block_index: 0,
            buffer_offset: 0,
            bytes: LOAD_BYTES as u16,
            address: self.source_address,
            active_lane_mask: u16::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VaUpdate {
    pub destination_va: u8,
    pub entries: [u16; 8],
}

pub fn plan_c220_load_va_issue(
    pc: u64,
    word: u32,
    source_address: u64,
    ub: &UbMemory,
) -> Result<C220LoadVaIssue, C220VectorError> {
    let instruction =
        C220LoadVaInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    if instruction.destination_va >= 8 {
        return Err(C220VectorError::InvalidVaRegister(
            instruction.destination_va,
        ));
    }
    ub.check_range(source_address, LOAD_BYTES)?;
    Ok(C220LoadVaIssue {
        pc,
        word,
        instruction,
        source_address,
    })
}

pub fn evaluate_c220_load_va(issue: C220LoadVaIssue, bytes: &[u8]) -> C220VaUpdate {
    let offset = usize::from(issue.instruction.high_half) * HALF_BYTES;
    let mut entries = [0_u16; 8];
    for (index, entry) in entries.iter_mut().enumerate() {
        let start = offset + index * 2;
        *entry = u16::from_le_bytes([bytes[start], bytes[start + 1]]);
    }
    C220VaUpdate {
        destination_va: issue.instruction.destination_va,
        entries,
    }
}
