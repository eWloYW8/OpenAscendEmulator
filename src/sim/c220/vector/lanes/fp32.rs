use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::ub::UbMemory;
use crate::numeric::fp32::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation, evaluate_fp32_value,
    evaluate_masked_fp32_lanes,
};
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::vector::{
    C220_VECTOR_TILE_BYTES, C220_VECTOR32_LANES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit,
    plan_c220_vector_arithmetic_read_accesses, store_data, vector_destination_address,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Fp32Step {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub source_0_address: u64,
    pub source_1_address: u64,
    pub destination_address: u64,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub iteration_masks: Vec<[u64; 4]>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220VectorStore>,
}

pub(crate) struct C220Fp32RepeatOutcome {
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<Fp32LaneOutcome>,
    pub stores: Vec<C220VectorStore>,
}

pub fn execute_c220_fp32_to_ub(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    ub: &mut UbMemory,
) -> Result<C220Fp32Step, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| hint.has_fp32_value_path())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let mut staged = (iteration_masks.len() > 1).then(|| ub.clone());
    let memory = if let Some(staged) = &mut staged {
        staged
    } else {
        &mut *ub
    };
    let mut source_0_bytes = Vec::new();
    let mut source_1_bytes = Vec::new();
    let mut lanes = Vec::new();
    let mut stores = Vec::new();
    let max_lanes = C220_VECTOR32_LANES * iteration_masks.len();
    stores
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, active_mask) in iteration_masks.iter().enumerate() {
        let repeat =
            evaluate_c220_fp32_repeat(hint, control, addresses, repeat_index, active_mask, memory)?;
        crate::sim::c220::vector::access::commit_vector_stores(memory, &repeat.stores)?;
        source_0_bytes.extend(repeat.source_0_bytes);
        source_1_bytes.extend(repeat.source_1_bytes);
        lanes.extend(repeat.lanes);
        stores.extend(repeat.stores);
    }
    if let Some(staged) = staged {
        *ub = staged;
    }
    Ok(C220Fp32Step {
        pc,
        word,
        hint,
        control,
        source_0_address: addresses.source_0,
        source_1_address: addresses.source_1,
        destination_address: addresses.destination,
        source_0_bytes,
        source_1_bytes,
        iteration_masks: iteration_masks.to_vec(),
        lanes,
        stores,
    })
}

pub(crate) fn evaluate_c220_fp32_repeat(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let accesses = plan_c220_vector_arithmetic_read_accesses(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        None,
    )?;
    evaluate_c220_fp32_repeat_with_accesses(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        &accesses,
        ub,
    )
}

pub(crate) fn evaluate_c220_fp32_repeat_with_accesses(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    accesses: &[C220VectorReadAccess],
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let mut source_0_bytes = vec![0; C220_VECTOR_TILE_BYTES];
    let mut source_1_bytes = vec![0; C220_VECTOR_TILE_BYTES];
    for access in accesses {
        let destination = if access.source_index == 0 {
            &mut source_0_bytes
        } else {
            &mut source_1_bytes
        };
        let offset = usize::from(access.buffer_offset);
        let bytes = usize::from(access.bytes);
        destination[offset..offset + bytes].copy_from_slice(&ub.read_known(access.address, bytes)?);
    }
    evaluate_c220_fp32_repeat_from_bytes(
        hint,
        control,
        addresses,
        repeat_index,
        active_mask,
        source_0_bytes,
        source_1_bytes,
        ub,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_c220_fp32_repeat_from_bytes(
    hint: C220VecArithmeticHint,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    repeat_index: usize,
    active_mask: &[u64; 4],
    source_0_bytes: Vec<u8>,
    source_1_bytes: Vec<u8>,
    ub: &UbMemory,
) -> Result<C220Fp32RepeatOutcome, C220VectorError> {
    let first = source_0_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let second = source_1_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    let lanes = evaluate_c220_fp32_lanes(hint, &first, &second, active_mask)?;
    let mut stores = Vec::new();
    for (lane_index, lane) in lanes.iter().enumerate() {
        if !lane.active {
            continue;
        }
        let address = vector_destination_address(control, addresses, repeat_index, lane_index)?;
        ub.check_range(address, 4)?;
        stores.push(C220VectorStore {
            repeat_index,
            lane_index,
            address,
            bank: C220UbBank::from_address(address),
            width_bytes: 4,
            data: store_data(lane.bits.to_le_bytes()),
        });
    }
    Ok(C220Fp32RepeatOutcome {
        source_0_bytes,
        source_1_bytes,
        lanes,
        stores,
    })
}

pub fn evaluate_c220_fp32_lanes(
    hint: C220VecArithmeticHint,
    first: &[u32],
    second: &[u32],
    iteration_mask: &[u64; 4],
) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
    if !hint.has_fp32_value_path() {
        return Err(Fp32VectorError::UnsupportedInstruction);
    }
    let rectify_result = matches!(
        hint.operation,
        C220VecArithmeticOperation::AddRectify | C220VecArithmeticOperation::SubtractRectify
    );
    let operation = match hint.operation {
        C220VecArithmeticOperation::Absolute => Fp32VectorOperation::Absolute,
        C220VecArithmeticOperation::Rectify => Fp32VectorOperation::Rectify,
        C220VecArithmeticOperation::Add | C220VecArithmeticOperation::AddRectify => {
            Fp32VectorOperation::Add
        }
        C220VecArithmeticOperation::Subtract | C220VecArithmeticOperation::SubtractRectify => {
            Fp32VectorOperation::Subtract
        }
        C220VecArithmeticOperation::Multiply => Fp32VectorOperation::Multiply,
        C220VecArithmeticOperation::Divide => Fp32VectorOperation::Divide,
        C220VecArithmeticOperation::Maximum => Fp32VectorOperation::Maximum,
        C220VecArithmeticOperation::Minimum => Fp32VectorOperation::Minimum,
        C220VecArithmeticOperation::Or
        | C220VecArithmeticOperation::And
        | C220VecArithmeticOperation::Not => {
            return Err(Fp32VectorError::UnsupportedInstruction);
        }
    };
    let mut lanes = evaluate_masked_fp32_lanes(
        operation,
        Fp32MaskLayout::C220Lane,
        first,
        second,
        iteration_mask,
    )?;
    if rectify_result {
        for lane in &mut lanes {
            if lane.active {
                lane.bits = evaluate_fp32_value(Fp32VectorOperation::Rectify, lane.bits, 0).bits;
            }
        }
    }
    Ok(lanes)
}
