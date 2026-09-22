use crate::isa::c220::vector::{C220VecArithmeticHint, C220VecArithmeticOperation};
use crate::memory::ub::UbMemory;
use crate::sim::c220::memory::C220UbBank;
use crate::sim::c220::numeric::fp16::C220Fp16Mode;
use crate::sim::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220VectorAddresses, C220VectorControl,
    C220VectorError, C220VectorReadAccess, C220VectorStore, check_repeat_limit,
    plan_c220_vector_arithmetic_read_accesses, vector_destination_address_for_width,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorArithmeticModes {
    pub integer_saturating: bool,
    pub fp16_mode: C220Fp16Mode,
    pub widen_s16: bool,
}

impl C220VectorArithmeticModes {
    pub const fn from_control_spr(value: u64) -> Self {
        Self {
            integer_saturating: value & (1 << 53) != 0,
            fp16_mode: C220Fp16Mode::from_control_spr(value),
            widen_s16: value & (1 << 52) != 0,
        }
    }

    pub const fn result_element_bytes(self, hint: C220VecArithmeticHint) -> Option<u8> {
        if self.widens(hint) {
            Some(4)
        } else {
            hint.modeled_element_bytes()
        }
    }

    pub const fn widens(self, hint: C220VecArithmeticHint) -> bool {
        self.widen_s16
            && hint.has_s16_value_path()
            && matches!(
                hint.operation,
                C220VecArithmeticOperation::Add
                    | C220VecArithmeticOperation::Subtract
                    | C220VecArithmeticOperation::Multiply
            )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220VectorArithmeticIssue {
    pub pc: u64,
    pub word: u32,
    pub hint: C220VecArithmeticHint,
    pub control: C220VectorControl,
    pub addresses: C220VectorAddresses,
    pub iteration_masks: Vec<[u64; 4]>,
    pub source_element_bytes: u8,
    pub result_element_bytes: u8,
    pub modes: C220VectorArithmeticModes,
    pub(crate) write_targets: Vec<C220VectorStore>,
}

impl C220VectorArithmeticIssue {
    pub fn read_accesses_for_repeat(
        &self,
        repeat_index: usize,
        lane_group: u8,
    ) -> Result<Vec<C220VectorReadAccess>, C220VectorError> {
        let mask = self
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorError::MissingMaskState)?;
        plan_c220_vector_arithmetic_read_accesses(
            self.hint,
            self.control,
            self.addresses,
            repeat_index,
            mask,
            (self.result_element_bytes == 2).then_some(lane_group),
        )
    }
}

pub fn plan_c220_vector_arithmetic_issue(
    pc: u64,
    word: u32,
    control: C220VectorControl,
    addresses: C220VectorAddresses,
    iteration_masks: &[[u64; 4]],
    modes: C220VectorArithmeticModes,
    ub: &UbMemory,
) -> Result<C220VectorArithmeticIssue, C220VectorError> {
    check_repeat_limit(iteration_masks.len())?;
    let hint = C220VecArithmeticHint::from_word(word)
        .filter(|hint| modes.result_element_bytes(*hint).is_some())
        .ok_or(C220VectorError::UnsupportedWord { pc, word })?;
    let source_element_bytes = hint
        .modeled_element_bytes()
        .expect("checked supported type");
    let result_element_bytes = modes
        .result_element_bytes(hint)
        .expect("checked supported type");
    let lane_count = C220_VECTOR_TILE_BYTES / usize::from(result_element_bytes);
    let mut effective_masks = iteration_masks.to_vec();
    if modes.widens(hint) {
        for mask in &mut effective_masks {
            mask[1..].fill(0);
        }
    }
    let mut write_targets = Vec::new();
    let max_lanes = lane_count * iteration_masks.len();
    write_targets
        .try_reserve_exact(max_lanes)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: max_lanes })?;
    for (repeat_index, mask) in effective_masks.iter().enumerate() {
        for access in plan_c220_vector_arithmetic_read_accesses(
            hint,
            control,
            addresses,
            repeat_index,
            mask,
            None,
        )? {
            ub.check_range(access.address, C220_VECTOR_BLOCK_BYTES)?;
        }
        for lane_index in 0..lane_count {
            if mask[lane_index / 64] & (1_u64 << (lane_index % 64)) == 0 {
                continue;
            }
            let address = vector_destination_address_for_width(
                control,
                addresses,
                repeat_index,
                lane_index,
                result_element_bytes,
            )?;
            ub.check_range(address, usize::from(result_element_bytes))?;
            write_targets.push(C220VectorStore {
                repeat_index,
                lane_index,
                address,
                bank: C220UbBank::from_address(address),
                width_bytes: result_element_bytes,
                data: [0; 8],
            });
        }
    }
    Ok(C220VectorArithmeticIssue {
        pc,
        word,
        hint,
        control,
        addresses,
        iteration_masks: effective_masks,
        source_element_bytes,
        result_element_bytes,
        modes,
        write_targets,
    })
}
