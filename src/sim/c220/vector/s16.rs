use super::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorStore, vector_destination_address_for_width,
};
use crate::architecture::c220::C220UbBank;
use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::ub::UbMemory;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct C220S16LaneOutcome {
    pub active: bool,
    pub bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct C220S16ValueInputs {
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
    pub mask: [u64; 4],
    pub saturating: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct C220S16WidenInputs {
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub repeat_index: usize,
    pub mask: [u64; 4],
}

pub(super) fn evaluate_c220_s16_widen_repeat_from_bytes(
    inputs: C220S16WidenInputs,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220S16LaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if !inputs.hint.has_s16_value_path()
        || !matches!(
            inputs.hint.operation,
            C220VecArithmeticOperation::Add
                | C220VecArithmeticOperation::Subtract
                | C220VecArithmeticOperation::Multiply
        )
    {
        return Err(C220VectorError::UnsupportedS16Operation);
    }
    for source in [source_0_bytes, source_1_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    let mut lanes = Vec::with_capacity(64);
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..64 {
        let active = inputs.mask[0] & (1_u64 << lane_index) != 0;
        if !active {
            lanes.push(C220S16LaneOutcome {
                active: false,
                bits: 0,
            });
            continue;
        }
        let offset = lane_index * 2;
        let first = i32::from(i16::from_le_bytes(
            source_0_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        ));
        let second = i32::from(i16::from_le_bytes(
            source_1_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        ));
        let value = match inputs.hint.operation {
            C220VecArithmeticOperation::Add => first + second,
            C220VecArithmeticOperation::Subtract => first - second,
            C220VecArithmeticOperation::Multiply => first * second,
            _ => unreachable!("widening operation was checked"),
        };
        let address = vector_destination_address_for_width(
            inputs.control,
            inputs.addresses,
            inputs.repeat_index,
            lane_index,
            4,
        )?;
        ub.check_range(address, 4)?;
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: super::store_data(value.to_le_bytes()),
        });
        lanes.push(C220S16LaneOutcome {
            active,
            bits: value as u32,
        });
    }
    Ok((lanes, stores))
}

pub(super) fn evaluate_c220_s16_repeat_from_bytes(
    inputs: C220S16ValueInputs,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220S16LaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if !inputs.hint.has_s16_value_path() && !inputs.hint.has_bitwise_b16_value_path() {
        return Err(C220VectorError::UnsupportedS16Operation);
    }
    if inputs.lane_group >= 2 {
        return Err(C220VectorError::InvalidLaneGroup(inputs.lane_group));
    }
    for source in [source_0_bytes, source_1_bytes] {
        if source.len() != C220_VECTOR_TILE_BYTES {
            return Err(C220VectorError::InvalidSourceTile {
                actual: source.len(),
                expected: C220_VECTOR_TILE_BYTES,
            });
        }
    }
    let mut lanes = Vec::with_capacity(C220_VECTOR_TILE_BYTES / 2);
    let mut stores = Vec::with_capacity(64);
    for lane_index in 0..C220_VECTOR_TILE_BYTES / 2 {
        let in_uop = inputs.lane_slice.map_or_else(
            || lane_index / 64 == usize::from(inputs.lane_group),
            |(first_lane, lane_count)| {
                lane_index >= first_lane && lane_index < first_lane + lane_count
            },
        );
        let active = in_uop && inputs.mask[lane_index / 64] & (1_u64 << (lane_index % 64)) != 0;
        if !active {
            lanes.push(C220S16LaneOutcome {
                active: false,
                bits: 0,
            });
            continue;
        }
        let offset = lane_index * 2;
        let first = i16::from_le_bytes(
            source_0_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        );
        let second = i16::from_le_bytes(
            source_1_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        );
        let value = match inputs.hint.operation {
            C220VecArithmeticOperation::Add if inputs.saturating => first.saturating_add(second),
            C220VecArithmeticOperation::Add => first.wrapping_add(second),
            C220VecArithmeticOperation::Subtract if inputs.saturating => {
                first.saturating_sub(second)
            }
            C220VecArithmeticOperation::Subtract => first.wrapping_sub(second),
            C220VecArithmeticOperation::AddRectify => {
                let sum = if inputs.saturating {
                    first.saturating_add(second)
                } else {
                    first.wrapping_add(second)
                };
                sum.max(0)
            }
            C220VecArithmeticOperation::SubtractRectify => {
                let difference = if inputs.saturating {
                    first.saturating_sub(second)
                } else {
                    first.wrapping_sub(second)
                };
                difference.max(0)
            }
            C220VecArithmeticOperation::Multiply if inputs.saturating => {
                first.saturating_mul(second)
            }
            C220VecArithmeticOperation::Multiply => first.wrapping_mul(second),
            C220VecArithmeticOperation::Maximum => first.max(second),
            C220VecArithmeticOperation::Minimum => first.min(second),
            C220VecArithmeticOperation::Absolute => first.wrapping_abs(),
            C220VecArithmeticOperation::Or => first | second,
            C220VecArithmeticOperation::And => first & second,
            C220VecArithmeticOperation::Not => !first,
            _ => unreachable!("S16 value path was checked"),
        };
        let address = vector_destination_address_for_width(
            inputs.control,
            inputs.addresses,
            inputs.repeat_index,
            lane_index,
            2,
        )?;
        ub.check_range(address, 2)?;
        let bytes = value.to_le_bytes();
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 2,
            data: super::store_data(bytes),
        });
        lanes.push(C220S16LaneOutcome {
            active,
            bits: u32::from(value as u16),
        });
    }
    Ok((lanes, stores))
}

#[cfg(test)]
mod tests;
