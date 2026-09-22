use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220_VECTOR32_LANES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorStore, vector_destination_address,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::vector) struct C220S32LaneOutcome {
    pub active: bool,
    pub bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::vector) struct C220S32ValueInputs {
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub repeat_index: usize,
    pub mask: [u64; 4],
    pub saturating: bool,
}

pub(in crate::sim::c220::vector) fn evaluate_c220_s32_repeat_from_bytes(
    inputs: C220S32ValueInputs,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220S32LaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if !inputs.hint.has_s32_value_path() {
        return Err(C220VectorError::UnsupportedS32Operation);
    }
    for source in [source_0_bytes, source_1_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    let mut lanes = Vec::with_capacity(C220_VECTOR32_LANES);
    let mut stores = Vec::new();
    for lane_index in 0..C220_VECTOR32_LANES {
        let active = inputs.mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220S32LaneOutcome {
                active: false,
                bits: 0,
            });
            continue;
        }
        let byte_offset = lane_index * 4;
        let first = u32::from_le_bytes(
            source_0_bytes[byte_offset..byte_offset + 4]
                .try_into()
                .expect("four-byte lane"),
        );
        let second = u32::from_le_bytes(
            source_1_bytes[byte_offset..byte_offset + 4]
                .try_into()
                .expect("four-byte lane"),
        );
        let bits = match inputs.hint.operation {
            C220VecArithmeticOperation::Add if inputs.saturating => {
                (first as i32).saturating_add(second as i32) as u32
            }
            C220VecArithmeticOperation::Add => first.wrapping_add(second),
            C220VecArithmeticOperation::Subtract if inputs.saturating => {
                (first as i32).saturating_sub(second as i32) as u32
            }
            C220VecArithmeticOperation::Subtract => first.wrapping_sub(second),
            C220VecArithmeticOperation::Multiply if inputs.saturating => {
                (first as i32).saturating_mul(second as i32) as u32
            }
            C220VecArithmeticOperation::Multiply => first.wrapping_mul(second),
            C220VecArithmeticOperation::Maximum => (first as i32).max(second as i32) as u32,
            C220VecArithmeticOperation::Minimum => (first as i32).min(second as i32) as u32,
            C220VecArithmeticOperation::Rectify => (first as i32).max(0) as u32,
            _ => unreachable!("S32 value path was checked"),
        };
        let address = vector_destination_address(
            inputs.control,
            inputs.addresses,
            inputs.repeat_index,
            lane_index,
        )?;
        ub.check_range(address, 4)?;
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: crate::sim::c220::vector::access::store_data(bits.to_le_bytes()),
        });
        lanes.push(C220S32LaneOutcome { active, bits });
    }
    Ok((lanes, stores))
}
