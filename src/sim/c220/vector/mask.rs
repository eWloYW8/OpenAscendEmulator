use super::{C220_VECTOR32_LANES, C220VectorError};
const MAX_VECTOR_REPEATS: usize = 4096;
pub(super) const C220_COUNT_MASK_CONTROL: u64 = 1 << 56;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorMaskState {
    pub control: u64,
    pub low: u64,
    pub high: u64,
}

pub fn decode_c220_fp32_mask(
    control: u64,
    mask0: u64,
    mask1: u64,
) -> Result<[u64; 4], C220VectorError> {
    decode_c220_tile_mask(control, mask0, mask1, C220_VECTOR32_LANES)
}

pub(crate) fn decode_c220_repeat_masks(
    mask_control: u64,
    mask0: u64,
    mask1: u64,
    lane_count: usize,
    encoded_repeat_count: u8,
) -> Result<Vec<[u64; 4]>, C220VectorError> {
    let mask_mode = mask_control & !((1 << 48) | (1 << 53) | (1 << 59));
    let repeat_count = match mask_mode {
        0 => u64::from(encoded_repeat_count),
        C220_COUNT_MASK_CONTROL => {
            if mask1 != 0 {
                return Err(C220VectorError::UnsupportedCountMaskHigh { high: mask1 });
            }
            mask0.div_ceil(lane_count as u64)
        }
        _ => {
            return Err(C220VectorError::UnsupportedMaskControl {
                control: mask_control,
            });
        }
    };
    if repeat_count > MAX_VECTOR_REPEATS as u64 {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeat_count,
            limit: MAX_VECTOR_REPEATS,
        });
    }
    let repeats = repeat_count as usize;
    let mut masks = Vec::new();
    masks
        .try_reserve_exact(repeats)
        .map_err(|_| C220VectorError::HostAllocationFailed { lanes: repeats })?;
    for repeat in 0..repeats {
        let mask = if mask_mode == 0 {
            [mask0, mask1, 0, 0]
        } else {
            let remaining = mask0.saturating_sub((repeat * lane_count) as u64);
            decode_c220_tile_mask(
                C220_COUNT_MASK_CONTROL,
                remaining.min(lane_count as u64),
                0,
                lane_count,
            )?
        };
        masks.push(mask);
    }
    Ok(masks)
}

pub(crate) fn decode_c220_tile_mask(
    control: u64,
    mask0: u64,
    mask1: u64,
    lane_count: usize,
) -> Result<[u64; 4], C220VectorError> {
    match control & !((1 << 48) | (1 << 53) | (1 << 59)) {
        0 => Ok([mask0, mask1, 0, 0]),
        C220_COUNT_MASK_CONTROL => {
            if mask1 != 0 {
                return Err(C220VectorError::UnsupportedCountMaskHigh { high: mask1 });
            }
            if mask0 > lane_count as u64 {
                return Err(C220VectorError::CountMaskExceedsTile { count: mask0 });
            }
            let low = if mask0 >= 64 {
                u64::MAX
            } else {
                (1_u64 << mask0) - 1
            };
            let high_count = mask0.saturating_sub(64);
            let high = if high_count == 64 {
                u64::MAX
            } else {
                (1_u64 << high_count) - 1
            };
            Ok([low, high, 0, 0])
        }
        _ => Err(C220VectorError::UnsupportedMaskControl { control }),
    }
}

pub(crate) fn check_repeat_limit(repeats: usize) -> Result<(), C220VectorError> {
    if repeats > MAX_VECTOR_REPEATS {
        return Err(C220VectorError::RepeatLimitExceeded {
            count: repeats as u64,
            limit: MAX_VECTOR_REPEATS,
        });
    }
    Ok(())
}
