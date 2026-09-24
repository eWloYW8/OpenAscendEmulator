use thiserror::Error;

mod fma;
pub use fma::evaluate_fp32_fused_multiply_add;

const SIGN_BIT: u32 = 0x8000_0000;
const ABS_MASK: u32 = 0x7fff_ffff;
const INFINITY_BITS: u32 = 0x7f80_0000;
const MAX_FINITE_BITS: u32 = 0x7f7f_ffff;
const CANONICAL_NAN_BITS: u32 = 0x7fff_ffff;
const MASK_WORDS: usize = 4;
const MAX_FP32_LANES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp32VectorOperation {
    Absolute,
    Rectify,
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp32MaskLayout {
    C220Lane,
    C310ByteStart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Fp32ValueStatus {
    pub overflow: bool,
    pub underflow: bool,
    pub nan_operand: bool,
    pub infinity_operand: bool,
    pub opposite_infinities: bool,
    pub zero_times_infinity: bool,
    pub division_by_zero: bool,
    pub indeterminate_division: bool,
    pub invalid: bool,
}

impl Fp32ValueStatus {
    pub const fn merge(self, other: Self) -> Self {
        Self {
            overflow: self.overflow || other.overflow,
            underflow: self.underflow || other.underflow,
            nan_operand: self.nan_operand || other.nan_operand,
            infinity_operand: self.infinity_operand || other.infinity_operand,
            opposite_infinities: self.opposite_infinities || other.opposite_infinities,
            zero_times_infinity: self.zero_times_infinity || other.zero_times_infinity,
            division_by_zero: self.division_by_zero || other.division_by_zero,
            indeterminate_division: self.indeterminate_division || other.indeterminate_division,
            invalid: self.invalid || other.invalid,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp32ValueOutcome {
    pub bits: u32,
    pub status: Fp32ValueStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp32LaneOutcome {
    pub active: bool,
    pub bits: u32,
    pub status: Option<Fp32ValueStatus>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp32WritebackPolicy {
    C220Masked { write_even_when_mask_clear: bool },
    C310WholeResult,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fp32WritebackOutcome {
    pub words: Vec<u32>,
    pub written: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum Fp32VectorError {
    #[error("FP32 source lengths differ: first has {first}, second has {second} lanes")]
    LengthMismatch { first: usize, second: usize },
    #[error("FP32 vector has {lanes} lanes, beyond the 256-bit mask's 64 FP32 lanes")]
    TooManyLanes { lanes: usize },
    #[error("FP32 destination has {destination} lanes but the result has {results}")]
    DestinationTooSmall { destination: usize, results: usize },
    #[error("word does not select a supported FP32 vector arithmetic path")]
    UnsupportedInstruction,
    #[error("cannot reserve memory for {lanes} FP32 lane results")]
    HostAllocationFailed { lanes: usize },
}

pub fn evaluate_fp32_value(
    operation: Fp32VectorOperation,
    first_bits: u32,
    second_bits: u32,
) -> Fp32ValueOutcome {
    if operation == Fp32VectorOperation::Absolute {
        let magnitude = first_bits & ABS_MASK;
        let status = Fp32ValueStatus {
            nan_operand: magnitude > INFINITY_BITS,
            infinity_operand: magnitude == INFINITY_BITS,
            ..Fp32ValueStatus::default()
        };
        return Fp32ValueOutcome {
            bits: if status.nan_operand {
                CANONICAL_NAN_BITS
            } else {
                magnitude
            },
            status,
        };
    }
    if operation == Fp32VectorOperation::Rectify {
        let magnitude = first_bits & ABS_MASK;
        let status = Fp32ValueStatus {
            nan_operand: magnitude > INFINITY_BITS,
            infinity_operand: magnitude == INFINITY_BITS,
            ..Fp32ValueStatus::default()
        };
        return Fp32ValueOutcome {
            bits: if status.nan_operand {
                CANONICAL_NAN_BITS
            } else if first_bits & SIGN_BIT != 0 {
                0
            } else {
                first_bits
            },
            status,
        };
    }
    if operation == Fp32VectorOperation::Multiply {
        return evaluate_fp32_product(first_bits, second_bits);
    }
    if operation == Fp32VectorOperation::Divide {
        return evaluate_fp32_quotient(first_bits, second_bits);
    }
    if matches!(
        operation,
        Fp32VectorOperation::Maximum | Fp32VectorOperation::Minimum
    ) {
        return evaluate_fp32_extremum(operation, first_bits, second_bits);
    }
    let second_bits = match operation {
        Fp32VectorOperation::Absolute | Fp32VectorOperation::Rectify => unreachable!(),
        Fp32VectorOperation::Add => second_bits,
        Fp32VectorOperation::Subtract => second_bits ^ SIGN_BIT,
        Fp32VectorOperation::Multiply
        | Fp32VectorOperation::Divide
        | Fp32VectorOperation::Maximum
        | Fp32VectorOperation::Minimum => unreachable!(),
    };
    let first_abs = first_bits & ABS_MASK;
    let second_abs = second_bits & ABS_MASK;
    let first_nan = first_abs > INFINITY_BITS;
    let second_nan = second_abs > INFINITY_BITS;
    let first_infinite = first_abs == INFINITY_BITS;
    let second_infinite = second_abs == INFINITY_BITS;
    let opposite_infinities =
        first_infinite && second_infinite && ((first_bits ^ second_bits) & SIGN_BIT) != 0;
    let mut status = Fp32ValueStatus {
        nan_operand: first_nan || second_nan,
        infinity_operand: first_infinite || second_infinite,
        opposite_infinities,
        ..Fp32ValueStatus::default()
    };

    if status.nan_operand || opposite_infinities {
        return Fp32ValueOutcome {
            bits: CANONICAL_NAN_BITS,
            status,
        };
    }
    if status.infinity_operand {
        let sign = if first_infinite {
            first_bits & SIGN_BIT
        } else {
            second_bits & SIGN_BIT
        };
        return Fp32ValueOutcome {
            bits: sign | INFINITY_BITS,
            status,
        };
    }

    let first = f32::from_bits(first_bits);
    let second = f32::from_bits(second_bits);
    let exact_sum = (first as f64) + (second as f64);
    let rounded_abs = exact_sum.abs() as f32;
    status.overflow = rounded_abs.is_infinite();
    status.underflow = exact_sum != 0.0 && exact_sum.abs() <= (f32::from_bits(1) as f64) * 0.5;
    if status.overflow {
        return Fp32ValueOutcome {
            bits: (first_bits & second_bits & SIGN_BIT) | INFINITY_BITS,
            status,
        };
    }

    let mut bits = (first + second).to_bits();
    if bits == SIGN_BIT && !(first_bits == SIGN_BIT && second_bits == SIGN_BIT) {
        bits = 0;
    }
    if bits & INFINITY_BITS == INFINITY_BITS {
        bits = (bits & SIGN_BIT) | MAX_FINITE_BITS;
    }
    Fp32ValueOutcome { bits, status }
}

fn evaluate_fp32_extremum(
    operation: Fp32VectorOperation,
    first_bits: u32,
    second_bits: u32,
) -> Fp32ValueOutcome {
    let first_abs = first_bits & ABS_MASK;
    let second_abs = second_bits & ABS_MASK;
    let status = Fp32ValueStatus {
        nan_operand: first_abs > INFINITY_BITS || second_abs > INFINITY_BITS,
        infinity_operand: first_abs == INFINITY_BITS || second_abs == INFINITY_BITS,
        ..Fp32ValueStatus::default()
    };
    if status.nan_operand {
        return Fp32ValueOutcome {
            bits: CANONICAL_NAN_BITS,
            status,
        };
    }
    let first = f32::from_bits(first_bits);
    let second = f32::from_bits(second_bits);
    let choose_first = if first == second {
        match operation {
            Fp32VectorOperation::Maximum => (first_bits & SIGN_BIT) <= (second_bits & SIGN_BIT),
            Fp32VectorOperation::Minimum => (first_bits & SIGN_BIT) >= (second_bits & SIGN_BIT),
            _ => unreachable!(),
        }
    } else {
        match operation {
            Fp32VectorOperation::Maximum => first > second,
            Fp32VectorOperation::Minimum => first < second,
            _ => unreachable!(),
        }
    };
    Fp32ValueOutcome {
        bits: if choose_first {
            first_bits
        } else {
            second_bits
        },
        status,
    }
}

fn evaluate_fp32_product(first_bits: u32, second_bits: u32) -> Fp32ValueOutcome {
    let first_abs = first_bits & ABS_MASK;
    let second_abs = second_bits & ABS_MASK;
    let first_infinite = first_abs == INFINITY_BITS;
    let second_infinite = second_abs == INFINITY_BITS;
    let mut status = Fp32ValueStatus {
        nan_operand: first_abs > INFINITY_BITS || second_abs > INFINITY_BITS,
        infinity_operand: first_infinite || second_infinite,
        zero_times_infinity: (first_infinite && second_abs == 0)
            || (second_infinite && first_abs == 0),
        ..Fp32ValueStatus::default()
    };
    if status.nan_operand || status.zero_times_infinity {
        return Fp32ValueOutcome {
            bits: CANONICAL_NAN_BITS,
            status,
        };
    }
    if status.infinity_operand {
        return Fp32ValueOutcome {
            bits: ((first_bits ^ second_bits) & SIGN_BIT) | INFINITY_BITS,
            status,
        };
    }
    let first = f32::from_bits(first_bits);
    let second = f32::from_bits(second_bits);
    let exact_product = (first as f64) * (second as f64);
    let half_min_subnormal = (f32::from_bits(1) as f64) * 0.5;
    if exact_product != 0.0 && exact_product.abs() <= half_min_subnormal {
        status.underflow = true;
        return Fp32ValueOutcome {
            bits: (first_bits ^ second_bits) & SIGN_BIT,
            status,
        };
    }
    let product = first * second;
    status.overflow = product.is_infinite();
    Fp32ValueOutcome {
        bits: product.to_bits(),
        status,
    }
}

fn evaluate_fp32_quotient(first_bits: u32, second_bits: u32) -> Fp32ValueOutcome {
    let first_abs = first_bits & ABS_MASK;
    let second_abs = second_bits & ABS_MASK;
    let first_infinite = first_abs == INFINITY_BITS;
    let second_infinite = second_abs == INFINITY_BITS;
    let zero_divisor = second_abs == 0;
    let indeterminate = (first_abs == 0 && zero_divisor) || (first_infinite && second_infinite);
    let mut status = Fp32ValueStatus {
        overflow: zero_divisor && first_abs < INFINITY_BITS,
        nan_operand: first_abs > INFINITY_BITS || second_abs > INFINITY_BITS,
        infinity_operand: first_infinite || second_infinite,
        division_by_zero: zero_divisor,
        indeterminate_division: indeterminate,
        ..Fp32ValueStatus::default()
    };
    if status.nan_operand || indeterminate {
        return Fp32ValueOutcome {
            bits: CANONICAL_NAN_BITS,
            status,
        };
    }
    let sign = (first_bits ^ second_bits) & SIGN_BIT;
    if zero_divisor {
        return Fp32ValueOutcome {
            bits: sign | INFINITY_BITS,
            status,
        };
    }
    if first_infinite {
        return Fp32ValueOutcome {
            bits: sign | INFINITY_BITS,
            status,
        };
    }
    if second_infinite {
        return Fp32ValueOutcome { bits: sign, status };
    }
    let first = f32::from_bits(first_bits);
    let second = f32::from_bits(second_bits);
    let quotient = first / second;
    let exact = (first as f64) / (second as f64);
    status.overflow = exact.abs() >= f32::MAX as f64;
    status.underflow = first_abs != 0 && quotient == 0.0;
    Fp32ValueOutcome {
        bits: quotient.to_bits(),
        status,
    }
}

pub fn evaluate_masked_fp32_lanes(
    operation: Fp32VectorOperation,
    layout: Fp32MaskLayout,
    first: &[u32],
    second: &[u32],
    mask_words: &[u64; MASK_WORDS],
) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
    if first.len() != second.len() {
        return Err(Fp32VectorError::LengthMismatch {
            first: first.len(),
            second: second.len(),
        });
    }
    if first.len() > MAX_FP32_LANES {
        return Err(Fp32VectorError::TooManyLanes { lanes: first.len() });
    }
    let mut outcomes = Vec::new();
    outcomes
        .try_reserve_exact(first.len())
        .map_err(|_| Fp32VectorError::HostAllocationFailed { lanes: first.len() })?;
    for (lane, (&first_bits, &second_bits)) in first.iter().zip(second).enumerate() {
        let bit = match layout {
            Fp32MaskLayout::C220Lane => lane,
            Fp32MaskLayout::C310ByteStart => lane * 4,
        };
        let active = (mask_words[bit / 64] >> (bit % 64)) & 1 != 0;
        if active {
            let value = evaluate_fp32_value(operation, first_bits, second_bits);
            outcomes.push(Fp32LaneOutcome {
                active,
                bits: value.bits,
                status: Some(value.status),
            });
        } else {
            outcomes.push(Fp32LaneOutcome {
                active,
                bits: 0,
                status: None,
            });
        }
    }
    Ok(outcomes)
}

pub fn apply_fp32_writeback(
    policy: Fp32WritebackPolicy,
    previous_destination: &[u32],
    lanes: &[Fp32LaneOutcome],
) -> Result<Fp32WritebackOutcome, Fp32VectorError> {
    if previous_destination.len() > MAX_FP32_LANES {
        return Err(Fp32VectorError::TooManyLanes {
            lanes: previous_destination.len(),
        });
    }
    if lanes.len() > MAX_FP32_LANES {
        return Err(Fp32VectorError::TooManyLanes { lanes: lanes.len() });
    }
    if previous_destination.len() < lanes.len() {
        return Err(Fp32VectorError::DestinationTooSmall {
            destination: previous_destination.len(),
            results: lanes.len(),
        });
    }
    let mut words = Vec::new();
    words
        .try_reserve_exact(previous_destination.len())
        .map_err(|_| Fp32VectorError::HostAllocationFailed {
            lanes: previous_destination.len(),
        })?;
    words.extend_from_slice(previous_destination);
    let mut written = Vec::new();
    written
        .try_reserve_exact(previous_destination.len())
        .map_err(|_| Fp32VectorError::HostAllocationFailed {
            lanes: previous_destination.len(),
        })?;
    written.resize(previous_destination.len(), false);
    for (index, lane) in lanes.iter().enumerate() {
        let should_write = match policy {
            Fp32WritebackPolicy::C220Masked {
                write_even_when_mask_clear,
            } => lane.active || write_even_when_mask_clear,
            Fp32WritebackPolicy::C310WholeResult => true,
        };
        if should_write {
            words[index] = lane.bits;
            written[index] = true;
        }
    }
    Ok(Fp32WritebackOutcome { words, written })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_add_and_sub_probe_values_follow_both_templates() {
        for index in 0..1024 {
            let index = index as f32;
            let x = index * 0.5 - 31.0;
            let y = 64.0 - index * 0.25;
            for (operation, expected) in [
                (Fp32VectorOperation::Add, x + y),
                (Fp32VectorOperation::Subtract, x - y),
            ] {
                let outcome = evaluate_fp32_value(operation, x.to_bits(), y.to_bits());
                assert_eq!(outcome.bits, expected.to_bits());
                assert_eq!(outcome.status, Fp32ValueStatus::default());
            }
        }
    }

    #[test]
    fn nan_infinity_overflow_and_signed_zero_match_handler_branches() {
        let add = Fp32VectorOperation::Add;
        let sub = Fp32VectorOperation::Subtract;
        let nan = evaluate_fp32_value(add, 0x7fc0_1234, 1.0_f32.to_bits());
        assert_eq!(nan.bits, CANONICAL_NAN_BITS);
        assert!(nan.status.nan_operand);
        let invalid = evaluate_fp32_value(add, INFINITY_BITS, SIGN_BIT | INFINITY_BITS);
        assert_eq!(invalid.bits, CANONICAL_NAN_BITS);
        assert!(!invalid.status.nan_operand);
        assert!(invalid.status.opposite_infinities);
        let positive_infinity = evaluate_fp32_value(add, INFINITY_BITS, 1.0_f32.to_bits());
        assert_eq!(positive_infinity.bits, INFINITY_BITS);
        assert!(positive_infinity.status.infinity_operand);
        let negative_overflow =
            evaluate_fp32_value(add, (-f32::MAX).to_bits(), (-f32::MAX).to_bits());
        assert_eq!(negative_overflow.bits, SIGN_BIT | INFINITY_BITS);
        assert!(negative_overflow.status.overflow);
        let max_threshold = evaluate_fp32_value(add, f32::MAX.to_bits(), 0);
        assert_eq!(max_threshold.bits, MAX_FINITE_BITS);
        assert!(!max_threshold.status.overflow);
        assert_eq!(evaluate_fp32_value(add, SIGN_BIT, SIGN_BIT).bits, SIGN_BIT);
        assert_eq!(evaluate_fp32_value(add, SIGN_BIT, 0).bits, 0);
        assert_eq!(evaluate_fp32_value(sub, 0, 0).bits, 0);
        assert_eq!(evaluate_fp32_value(sub, SIGN_BIT, 0).bits, SIGN_BIT);
    }

    #[test]
    fn multiply_canonicalizes_nan_and_preserves_finite_sign_and_rounding() {
        let multiply = Fp32VectorOperation::Multiply;
        for (first, second, expected) in [
            (0, INFINITY_BITS, CANONICAL_NAN_BITS),
            (SIGN_BIT, SIGN_BIT | INFINITY_BITS, CANONICAL_NAN_BITS),
            (0x7fc0_1234, 1.0_f32.to_bits(), CANONICAL_NAN_BITS),
            (0xff80_0001, 0, CANONICAL_NAN_BITS),
            (
                SIGN_BIT | INFINITY_BITS,
                (-2.0_f32).to_bits(),
                INFINITY_BITS,
            ),
            (SIGN_BIT, 2.0_f32.to_bits(), SIGN_BIT),
            (f32::MAX.to_bits(), 2.0_f32.to_bits(), INFINITY_BITS),
            (1, 1.0_f32.to_bits(), 1),
            (0x0080_0000, 0.5_f32.to_bits(), 0x0040_0000),
            (1, 0.5_f32.to_bits(), 0),
            (0x3f80_0001, 0x3f7f_ffff, 0x3f80_0000),
        ] {
            assert_eq!(evaluate_fp32_value(multiply, first, second).bits, expected);
        }
        assert!(
            evaluate_fp32_value(multiply, 0, INFINITY_BITS)
                .status
                .zero_times_infinity
        );
        assert!(
            evaluate_fp32_value(multiply, f32::MAX.to_bits(), 2.0_f32.to_bits())
                .status
                .overflow
        );
        assert!(
            !evaluate_fp32_value(multiply, 0x0080_0000, 0.5_f32.to_bits())
                .status
                .underflow
        );
        let tiny = evaluate_fp32_value(multiply, 1, 0.5_f32.to_bits());
        assert_eq!(tiny.bits, 0);
        assert!(tiny.status.underflow);
    }

    #[test]
    fn divide_handles_finite_values_and_exceptional_operands() {
        let divide = Fp32VectorOperation::Divide;
        for (first, second, expected) in [
            (3.0_f32.to_bits(), 2.0_f32.to_bits(), 1.5_f32.to_bits()),
            (0x7fc0_1234, 1.0_f32.to_bits(), CANONICAL_NAN_BITS),
            (0, 0, CANONICAL_NAN_BITS),
            (INFINITY_BITS, INFINITY_BITS, CANONICAL_NAN_BITS),
            ((-3.0_f32).to_bits(), 0, SIGN_BIT | INFINITY_BITS),
            (SIGN_BIT, 2.0_f32.to_bits(), SIGN_BIT),
            (1.0_f32.to_bits(), INFINITY_BITS, 0),
            (1, f32::MAX.to_bits(), 0),
        ] {
            assert_eq!(evaluate_fp32_value(divide, first, second).bits, expected);
        }
        let zero = evaluate_fp32_value(divide, 0, 0);
        assert!(zero.status.division_by_zero);
        assert!(zero.status.indeterminate_division);
        assert!(zero.status.overflow);
        let tiny = evaluate_fp32_value(divide, 1, f32::MAX.to_bits());
        assert!(tiny.status.underflow);
    }

    #[test]
    fn isolated_camodel_edge_oracle_words_match_both_architectures() {
        for (first, second, expected_add, expected_sub) in [
            (0x8000_0000, 0x8000_0000, 0x8000_0000, 0x0000_0000),
            (0x8000_0000, 0x0000_0000, 0x0000_0000, 0x8000_0000),
            (0x7f80_0000, 0xff80_0000, 0x7fff_ffff, 0x7f80_0000),
            (0x7f80_0001, 0x3f80_0000, 0x7fff_ffff, 0x7fff_ffff),
            (0x7f7f_ffff, 0x0000_0000, 0x7f7f_ffff, 0x7f7f_ffff),
            (0x7f7f_ffff, 0x7f7f_ffff, 0x7f80_0000, 0x0000_0000),
            (0x0000_0001, 0x0000_0001, 0x0000_0002, 0x0000_0000),
            (0x0080_0000, 0x0000_0001, 0x0080_0001, 0x007f_ffff),
        ] {
            assert_eq!(
                evaluate_fp32_value(Fp32VectorOperation::Add, first, second).bits,
                expected_add
            );
            assert_eq!(
                evaluate_fp32_value(Fp32VectorOperation::Subtract, first, second).bits,
                expected_sub
            );
        }
    }

    #[test]
    fn two_architecture_mask_bit_layouts_zero_inactive_lanes() {
        let first = [1.0_f32.to_bits(), 2.0_f32.to_bits(), 3.0_f32.to_bits()];
        let second = [4.0_f32.to_bits(), 5.0_f32.to_bits(), 6.0_f32.to_bits()];
        let c220 = evaluate_masked_fp32_lanes(
            Fp32VectorOperation::Add,
            Fp32MaskLayout::C220Lane,
            &first,
            &second,
            &[0b101, 0, 0, 0],
        )
        .unwrap();
        let c310 = evaluate_masked_fp32_lanes(
            Fp32VectorOperation::Subtract,
            Fp32MaskLayout::C310ByteStart,
            &first,
            &second,
            &[0x101, 0, 0, 0],
        )
        .unwrap();
        assert_eq!(
            c220.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
            [5.0_f32.to_bits(), 0, 9.0_f32.to_bits()]
        );
        assert_eq!(
            c310.iter().map(|lane| lane.bits).collect::<Vec<_>>(),
            [(-3.0_f32).to_bits(), 0, (-3.0_f32).to_bits()]
        );
        assert_eq!(c220[1].status, None);
        assert_eq!(c310[1].status, None);
    }

    #[test]
    fn length_and_lane_limit_errors_are_explicit() {
        assert_eq!(
            evaluate_masked_fp32_lanes(
                Fp32VectorOperation::Add,
                Fp32MaskLayout::C220Lane,
                &[0],
                &[],
                &[0; 4]
            ),
            Err(Fp32VectorError::LengthMismatch {
                first: 1,
                second: 0
            })
        );
        assert_eq!(
            evaluate_masked_fp32_lanes(
                Fp32VectorOperation::Add,
                Fp32MaskLayout::C310ByteStart,
                &[0; 65],
                &[0; 65],
                &[0; 4]
            ),
            Err(Fp32VectorError::TooManyLanes { lanes: 65 })
        );
    }

    #[test]
    fn c220_preserves_masked_destination_and_optional_override_writes_zero() {
        let lanes = evaluate_masked_fp32_lanes(
            Fp32VectorOperation::Add,
            Fp32MaskLayout::C220Lane,
            &[1.0_f32.to_bits(), 2.0_f32.to_bits()],
            &[3.0_f32.to_bits(), 4.0_f32.to_bits()],
            &[1, 0, 0, 0],
        )
        .unwrap();
        let previous = [9.0_f32.to_bits(), 10.0_f32.to_bits(), 11.0_f32.to_bits()];
        let masked = apply_fp32_writeback(
            Fp32WritebackPolicy::C220Masked {
                write_even_when_mask_clear: false,
            },
            &previous,
            &lanes,
        )
        .unwrap();
        assert_eq!(masked.words, [4.0_f32.to_bits(), previous[1], previous[2]]);
        assert_eq!(masked.written, [true, false, false]);
        let forced = apply_fp32_writeback(
            Fp32WritebackPolicy::C220Masked {
                write_even_when_mask_clear: true,
            },
            &previous,
            &lanes,
        )
        .unwrap();
        assert_eq!(forced.words, [4.0_f32.to_bits(), 0, previous[2]]);
        assert_eq!(forced.written, [true, true, false]);
    }

    #[test]
    fn c310_overwrites_inactive_lane_with_zero_and_preserves_unreached_tail() {
        let lanes = evaluate_masked_fp32_lanes(
            Fp32VectorOperation::Subtract,
            Fp32MaskLayout::C310ByteStart,
            &[1.0_f32.to_bits(), 2.0_f32.to_bits()],
            &[3.0_f32.to_bits(), 4.0_f32.to_bits()],
            &[1, 0, 0, 0],
        )
        .unwrap();
        let previous = [9.0_f32.to_bits(), 10.0_f32.to_bits(), 11.0_f32.to_bits()];
        let written =
            apply_fp32_writeback(Fp32WritebackPolicy::C310WholeResult, &previous, &lanes).unwrap();
        assert_eq!(written.words, [(-2.0_f32).to_bits(), 0, previous[2]]);
        assert_eq!(written.written, [true, true, false]);
        assert_eq!(
            apply_fp32_writeback(Fp32WritebackPolicy::C310WholeResult, &[], &lanes),
            Err(Fp32VectorError::DestinationTooSmall {
                destination: 0,
                results: 2
            })
        );
    }
}
