use crate::isa::c220::vector::scalar::C220VectorScalarOperation;
use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::numeric::fp16::{
    C220Fp16Mode, C220Fp16Outcome, C220Fp16Status, evaluate_c220_fp16, evaluate_c220_fp16_abs,
    evaluate_c220_fp16_relu,
};
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl, C220VectorError,
    C220VectorStore, vector_destination_address_for_width,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::vector) struct C220F16LaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub status: Option<C220Fp16Status>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::sim::c220::vector) struct C220F16ValueInputs {
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub repeat_index: usize,
    pub lane_group: u8,
    pub lane_slice: Option<(usize, usize)>,
    pub mask: [u64; 4],
    pub mode: C220Fp16Mode,
}

pub(in crate::sim::c220::vector) fn evaluate_c220_f16_repeat_from_bytes(
    inputs: C220F16ValueInputs,
    source_0_bytes: &[u8],
    source_1_bytes: &[u8],
    ub: &UbMemory,
) -> Result<(Vec<C220F16LaneOutcome>, Vec<C220VectorStore>), C220VectorError> {
    if !inputs.hint.has_f16_value_path() {
        return Err(C220VectorError::UnsupportedF16Operation);
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
            lanes.push(C220F16LaneOutcome {
                active: false,
                bits: 0,
                status: None,
            });
            continue;
        }
        let offset = lane_index * 2;
        let first = u16::from_le_bytes(
            source_0_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        );
        let second = u16::from_le_bytes(
            source_1_bytes[offset..offset + 2]
                .try_into()
                .expect("two-byte lane"),
        );
        let outcome = evaluate_lane(inputs.hint.operation, first, second, inputs.mode);
        let address = vector_destination_address_for_width(
            inputs.control,
            inputs.addresses,
            inputs.repeat_index,
            lane_index,
            2,
        )?;
        ub.check_range(address, 2)?;
        let bytes = outcome.bits.to_le_bytes();
        stores.push(C220VectorStore {
            repeat_index: inputs.repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 2,
            data: crate::sim::c220::vector::access::store_data(bytes),
        });
        lanes.push(C220F16LaneOutcome {
            active,
            bits: u32::from(outcome.bits),
            status: Some(outcome.status),
        });
    }
    Ok((lanes, stores))
}

fn evaluate_lane(
    operation: C220VecArithmeticOperation,
    first: u16,
    second: u16,
    mode: C220Fp16Mode,
) -> C220Fp16Outcome {
    if operation == C220VecArithmeticOperation::Absolute {
        return evaluate_c220_fp16_abs(first, mode);
    }
    if operation == C220VecArithmeticOperation::Rectify {
        return evaluate_c220_fp16_relu(first, mode);
    }
    let rectify_result = matches!(
        operation,
        C220VecArithmeticOperation::AddRectify | C220VecArithmeticOperation::SubtractRectify
    );
    let mapped = match operation {
        C220VecArithmeticOperation::Add
        | C220VecArithmeticOperation::Subtract
        | C220VecArithmeticOperation::AddRectify
        | C220VecArithmeticOperation::SubtractRectify => C220VectorScalarOperation::Add,
        C220VecArithmeticOperation::Multiply => C220VectorScalarOperation::Multiply,
        C220VecArithmeticOperation::Maximum => C220VectorScalarOperation::Maximum,
        C220VecArithmeticOperation::Minimum => C220VectorScalarOperation::Minimum,
        _ => unreachable!("F16 value path was checked"),
    };
    let second = if matches!(
        operation,
        C220VecArithmeticOperation::Subtract | C220VecArithmeticOperation::SubtractRectify
    ) {
        second ^ 0x8000
    } else {
        second
    };
    let mut result = evaluate_c220_fp16(mapped, first, second, mode);
    if rectify_result {
        result.bits = evaluate_c220_fp16_relu(result.bits, mode).bits;
    }
    result
}

#[cfg(test)]
mod tests;
