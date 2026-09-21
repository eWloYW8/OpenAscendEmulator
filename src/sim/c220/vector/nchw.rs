use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::{C220NchwElement, C220NchwInstruction};
use crate::memory::ub::UbMemory;
use crate::sim::c220::va::C220VaRegisters;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220VectorError, C220VectorReadAccess, C220VectorStore,
};

const ROWS: usize = 16;
const TILE_BYTES: usize = ROWS * C220_VECTOR_BLOCK_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220NchwControl {
    pub repeat_count: u8,
    pub destination_repeat_stride: u16,
    pub source_repeat_stride: u16,
}

impl C220NchwControl {
    pub const fn decode(value: u64) -> Self {
        Self {
            repeat_count: (value >> 56) as u8,
            destination_repeat_stride: value as u16,
            source_repeat_stride: (value >> 16) as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220NchwRows {
    pub source: [u64; ROWS],
    pub destination: [u64; ROWS],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220NchwIssue {
    pub pc: u64,
    pub word: u32,
    pub instruction: C220NchwInstruction,
    pub control: C220NchwControl,
    pub rows: Vec<C220NchwRows>,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220NchwIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let rows = self
            .rows
            .get(repeat_index)
            .ok_or(C220VectorError::InvalidRepeatIndex(repeat_index))?;
        let active_lane_mask = match self.instruction.element {
            C220NchwElement::Byte | C220NchwElement::Half => u16::MAX,
            C220NchwElement::Word => 0xff,
        };
        Ok(rows
            .source
            .iter()
            .enumerate()
            .map(|(block_index, &address)| C220VectorReadAccess {
                source_index: 0,
                block_index: block_index as u8,
                buffer_offset: (block_index * C220_VECTOR_BLOCK_BYTES) as u16,
                bytes: C220_VECTOR_BLOCK_BYTES as u16,
                address,
                active_lane_mask,
            })
            .collect())
    }
}

pub fn plan_c220_nchw_issue(
    pc: u64,
    word: u32,
    control_value: u64,
    va: &C220VaRegisters,
    ub: &UbMemory,
) -> Result<C220NchwIssue, C220VectorError> {
    let instruction =
        C220NchwInstruction::decode(word).ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let control = C220NchwControl::decode(control_value);
    if control.repeat_count == 0 {
        return Ok(C220NchwIssue {
            pc,
            word,
            instruction,
            control,
            rows: Vec::new(),
            write_targets: Vec::new(),
        });
    }
    let source_entries = read_table(va, instruction.source_va)?;
    let destination_entries = read_table(va, instruction.destination_va)?;
    let mut rows = Vec::with_capacity(usize::from(control.repeat_count));
    let mut write_targets = Vec::new();
    let zero_tile = [0_u8; TILE_BYTES];
    for repeat_index in 0..usize::from(control.repeat_count) {
        let source = advance_table(source_entries, repeat_index, control.source_repeat_stride);
        let destination = advance_table(
            destination_entries,
            repeat_index,
            control.destination_repeat_stride,
        );
        for &address in &source {
            ub.check_range(address, C220_VECTOR_BLOCK_BYTES)?;
        }
        let repeat_rows = C220NchwRows {
            source,
            destination,
        };
        let (_, planned) =
            evaluate_c220_nchw_repeat(instruction, repeat_index, repeat_rows, &zero_tile)?;
        for store in &planned {
            ub.check_range(store.address, usize::from(store.width_bytes))?;
        }
        write_targets.extend(planned);
        rows.push(repeat_rows);
    }
    Ok(C220NchwIssue {
        pc,
        word,
        instruction,
        control,
        rows,
        write_targets,
    })
}

fn read_table(va: &C220VaRegisters, start: u8) -> Result<[u16; ROWS], C220VectorError> {
    let mut entries = [0_u16; ROWS];
    for (index, entry) in entries.iter_mut().enumerate() {
        let register = start + (index / 8) as u8;
        let slot = (index % 8) as u8;
        *entry = va
            .entry(register, slot)
            .ok_or(C220VectorError::MissingVaEntry {
                register,
                index: slot,
            })?;
    }
    Ok(entries)
}

fn advance_table(entries: [u16; ROWS], repeat_index: usize, stride: u16) -> [u64; ROWS] {
    entries.map(|entry| {
        let block = (usize::from(entry) + (repeat_index + 1) * usize::from(stride)) & 0x1fff;
        (block * C220_VECTOR_BLOCK_BYTES) as u64
    })
}

pub fn evaluate_c220_nchw_repeat(
    instruction: C220NchwInstruction,
    repeat_index: usize,
    rows: C220NchwRows,
    source_bytes: &[u8],
) -> Result<(Vec<u32>, Vec<C220VectorStore>), C220VectorError> {
    if source_bytes.len() != TILE_BYTES {
        return Err(C220VectorError::InvalidSourceTile {
            actual: source_bytes.len(),
            expected: TILE_BYTES,
        });
    }
    let (output_rows, columns, width) = match instruction.element {
        C220NchwElement::Byte => (16, 16, 1),
        C220NchwElement::Half => (16, 16, 2),
        C220NchwElement::Word => (8, 16, 4),
    };
    let mut values = Vec::with_capacity(output_rows * columns);
    let mut stores = Vec::with_capacity(output_rows * columns);
    for row in 0..output_rows {
        for column in 0..columns {
            let source_offset = column * C220_VECTOR_BLOCK_BYTES
                + match instruction.element {
                    C220NchwElement::Byte => usize::from(instruction.source_high) * 16 + row,
                    C220NchwElement::Half => row * 2,
                    C220NchwElement::Word => row * 4,
                };
            let (address, half, lane_index) = match instruction.element {
                C220NchwElement::Byte => (
                    rows.destination[row]
                        + (usize::from(instruction.destination_high) * 16 + column) as u64,
                    row / 8,
                    (row % 8) * 32 + usize::from(instruction.destination_high) * 16 + column,
                ),
                C220NchwElement::Half => (
                    rows.destination[row] + (column * 2) as u64,
                    row / 8,
                    (row % 8) * 16 + column,
                ),
                C220NchwElement::Word => (
                    rows.destination[row * 2 + column / 8] + ((column % 8) * 4) as u64,
                    row / 4,
                    (row % 4) * 16 + column,
                ),
            };
            let mut data = [0_u8; 4];
            data[..width].copy_from_slice(&source_bytes[source_offset..source_offset + width]);
            values.push(u32::from_le_bytes(data));
            stores.push(C220VectorStore {
                repeat_index: 2 * repeat_index + half,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: width as u8,
                data,
            });
        }
    }
    Ok((values, stores))
}
