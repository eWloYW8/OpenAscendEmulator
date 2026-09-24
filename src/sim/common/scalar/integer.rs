use super::*;
use crate::isa::scalar::ScalarBitCountOperation;

impl ScalarMachine {
    pub fn execute_word(&mut self, pc: u64, word: u32) -> Result<ScalarStep, ScalarMachineError> {
        let hint = ScalarInstruction::from_word(self.architecture, word)
            .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
        let (
            destination_register,
            source_register,
            source_value,
            second_source_register,
            second_source_value,
            value,
            signed_overflow,
            spr2,
        ) = match hint {
            ScalarInstruction::ScalarMoveX8Immediate {
                destination_register,
                encoded_immediate,
                ..
            } => (
                destination_register,
                None,
                None,
                None,
                None,
                u64::from(encoded_immediate) * 8,
                false,
                self.spr2,
            ),
            ScalarInstruction::ScalarKey0 {
                operation,
                dtype_field,
                destination_register,
                first_source_register,
                second_source_register,
                ..
            } => {
                let Some(&first_source_value) = self.xregs.get(usize::from(first_source_register))
                else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&second_source_value) =
                    self.xregs.get(usize::from(second_source_register))
                else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let (value, signed_overflow) = match (operation, dtype_field) {
                    (ScalarKey0Operation::Add, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_add(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::Subtract, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_sub(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::Multiply, 0) => {
                        let (value, overflow) =
                            (first_source_value as i64).overflowing_mul(second_source_value as i64);
                        (value as u64, overflow)
                    }
                    (ScalarKey0Operation::MultiplyAdd, 0) => {
                        let (product, multiply_overflow) =
                            (first_source_value as i64).overflowing_mul(second_source_value as i64);
                        let (value, add_overflow) = (self.xregs[usize::from(destination_register)]
                            as i64)
                            .overflowing_add(product);
                        (value as u64, multiply_overflow || add_overflow)
                    }
                    (ScalarKey0Operation::Divide, 0) => {
                        let dividend = first_source_value as i64;
                        let divisor = second_source_value as i64;
                        let quotient = if divisor == 0 {
                            if dividend < 0 { i64::MIN } else { i64::MAX }
                        } else if dividend == i64::MIN && divisor == -1 {
                            i64::MIN
                        } else {
                            dividend / divisor
                        };
                        (quotient as u64, false)
                    }
                    (ScalarKey0Operation::Divide, 1) => (
                        first_source_value
                            .checked_div(second_source_value)
                            .unwrap_or(u64::MAX),
                        false,
                    ),
                    (ScalarKey0Operation::Remainder, 0) => {
                        let dividend = first_source_value as i64;
                        let divisor = second_source_value as i64;
                        let remainder = if divisor == 0 {
                            dividend
                        } else if dividend == i64::MIN && divisor == -1 {
                            0
                        } else {
                            dividend % divisor
                        };
                        (remainder as u64, false)
                    }
                    (ScalarKey0Operation::Remainder, 1) => (
                        first_source_value
                            .checked_rem(second_source_value)
                            .unwrap_or(first_source_value),
                        false,
                    ),
                    (ScalarKey0Operation::Minimum, 0) => (
                        (first_source_value as i64).min(second_source_value as i64) as u64,
                        false,
                    ),
                    (ScalarKey0Operation::Maximum, 0) => (
                        (first_source_value as i64).max(second_source_value as i64) as u64,
                        false,
                    ),
                    (ScalarKey0Operation::And, 3) => {
                        (first_source_value & second_source_value, false)
                    }
                    (ScalarKey0Operation::Or, 3) => {
                        (first_source_value | second_source_value, false)
                    }
                    (ScalarKey0Operation::Xor, 3) => {
                        (first_source_value ^ second_source_value, false)
                    }
                    _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
                };
                (
                    destination_register,
                    Some(first_source_register),
                    Some(first_source_value),
                    Some(second_source_register),
                    Some(second_source_value),
                    value,
                    signed_overflow,
                    update_overflow_spr2(self.spr2, pc, signed_overflow),
                )
            }
            ScalarInstruction::ScalarKey2ShiftLeft {
                dtype_field,
                destination_register,
                count_register,
                encoded_immediate,
                ..
            } => {
                if dtype_field != 3 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let prior = self.xregs[usize::from(destination_register)];
                let (second_source_register, second_source_value, shift) =
                    if let Some(register) = count_register {
                        let count_value = self.xregs[usize::from(register)];
                        (
                            Some(register),
                            Some(count_value),
                            (count_value & 0x3f) as u32,
                        )
                    } else {
                        (None, None, u32::from(encoded_immediate))
                    };
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    second_source_register,
                    second_source_value,
                    prior << shift,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2ShiftRight {
                dtype_field,
                destination_register,
                count_register,
                encoded_immediate,
                ..
            } => {
                if !matches!(dtype_field, 0 | 1) {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let prior = self.xregs[usize::from(destination_register)];
                let (second_source_register, second_source_value, shift) =
                    if let Some(register) = count_register {
                        let count_value = self.xregs[usize::from(register)];
                        (
                            Some(register),
                            Some(count_value),
                            (count_value & 0x3f) as u32,
                        )
                    } else {
                        (None, None, u32::from(encoded_immediate))
                    };
                let value = if dtype_field == 0 {
                    ((prior as i64) >> shift) as u64
                } else {
                    prior >> shift
                };
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    second_source_register,
                    second_source_value,
                    value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2BitCount {
                operation,
                destination_register,
                source_register,
            } => {
                let source_value = self.xregs[usize::from(source_register)];
                let value = match operation {
                    ScalarBitCountOperation::Zeros => u64::from(source_value.count_zeros()),
                    ScalarBitCountOperation::Ones => u64::from(source_value.count_ones()),
                    ScalarBitCountOperation::LeadingZeros => {
                        u64::from(source_value.leading_zeros())
                    }
                    ScalarBitCountOperation::LeadingSignBits => {
                        if source_value == 0 || source_value == u64::MAX {
                            u64::MAX
                        } else if source_value >> 63 != 0 {
                            u64::from(source_value.leading_ones() - 1)
                        } else {
                            u64::from(source_value.leading_zeros() - 1)
                        }
                    }
                };
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2FindFirst {
                destination_register,
                source_register,
                find_set,
            } => {
                let source_value = self.xregs[usize::from(source_register)];
                let index = if find_set {
                    source_value.trailing_zeros()
                } else {
                    source_value.trailing_ones()
                };
                let value = if index == u64::BITS {
                    u64::MAX
                } else {
                    u64::from(index)
                };
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2MoveRegister {
                destination_register,
                source_register,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2Negate {
                dtype_field,
                destination_register,
                source_register,
                ..
            } => {
                if dtype_field != 0 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let signed_overflow = source_value == i64::MIN as u64;
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value.wrapping_neg(),
                    signed_overflow,
                    update_neg_overflow_spr2(self.architecture, self.spr2, pc, signed_overflow),
                )
            }
            ScalarInstruction::ScalarKey2Absolute {
                dtype_field,
                destination_register,
                source_register,
            } => {
                if dtype_field != 0 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    if (source_value as i64) < 0 {
                        source_value.wrapping_neg()
                    } else {
                        source_value
                    },
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2IntegerSqrt {
                dtype_field,
                destination_register,
                source_register,
            } => {
                if dtype_field != 0 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    (source_value as i64).unsigned_abs().isqrt(),
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2BitwiseNot {
                dtype_field,
                destination_register,
                source_register,
            } => {
                if dtype_field != 3 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    !source_value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2ZeroExtend {
                width,
                destination_register,
                source_register,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    source_value & width.mask(),
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2SignExtend {
                width_bits,
                destination_register,
                source_register,
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if self.xregs.get(usize::from(destination_register)).is_none() {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let shift = 64 - u32::from(width_bits);
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    (((source_value << shift) as i64) >> shift) as u64,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2Insert {
                destination_register,
                source_register,
                least_significant_bit,
                width_bits,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                if u16::from(least_significant_bit) + u16::from(width_bits) > 64 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let mask = ((1_u64 << width_bits) - 1) << least_significant_bit;
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    (prior & !mask) | ((source_value << least_significant_bit) & mask),
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2InsertImmediate {
                destination_register,
                position,
                immediate,
                extended,
                ..
            } => {
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let width = if extended {
                    8
                } else {
                    (u8::BITS - immediate.leading_zeros()).max(1)
                };
                if u32::from(position) + width > 64 {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                }
                let mask = ((1_u64 << width) - 1) << position;
                (
                    destination_register,
                    Some(destination_register),
                    Some(prior),
                    None,
                    None,
                    (prior & !mask) | ((u64::from(immediate) << position) & mask),
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey2BitSet {
                destination_register,
                source_register,
                set_bit,
                ..
            } => {
                let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let Some(&prior) = self.xregs.get(usize::from(destination_register)) else {
                    return Err(ScalarMachineError::UnsupportedWord { pc, word });
                };
                let mask = 1_u64 << (source_value & 63);
                (
                    destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    if set_bit { prior | mask } else { prior & !mask },
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey7 {
                operation,
                destination_register,
                encoded_immediate,
                halfword_lane,
                ..
            } => {
                let prior = self.xregs[usize::from(destination_register)];
                let value = match operation {
                    ScalarKey7Operation::MoveImmediate => u64::from(encoded_immediate),
                    ScalarKey7Operation::MoveKeep => {
                        let lane = halfword_lane.expect("decoder supplies MOVK lane");
                        let shift = u32::from(lane) * 16;
                        (prior & !(0xffff_u64 << shift)) | (u64::from(encoded_immediate) << shift)
                    }
                };
                let source = (operation == ScalarKey7Operation::MoveKeep)
                    .then_some((destination_register, prior));
                (
                    destination_register,
                    source.map(|(register, _)| register),
                    source.map(|(_, value)| value),
                    None,
                    None,
                    value,
                    false,
                    self.spr2,
                )
            }
            ScalarInstruction::ScalarKey8 {
                source_register,
                destination_register: Some(_),
                ..
            } => {
                let source_value = self.xregs[usize::from(source_register)];
                let result = evaluate_scalar_integer_immediate(hint, source_value, pc, self.spr2)?;
                (
                    result.destination_register,
                    Some(source_register),
                    Some(source_value),
                    None,
                    None,
                    result.value,
                    result.signed_overflow,
                    result.spr2,
                )
            }
            _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
        };
        let prior_destination_value = self.xregs[usize::from(destination_register)];
        let prior_spr2 = self.spr2;
        self.xregs[usize::from(destination_register)] = value;
        self.spr2 = spr2;
        Ok(ScalarStep {
            pc,
            word,
            destination_register,
            prior_destination_value,
            source_register,
            source_value,
            second_source_register,
            second_source_value,
            value,
            signed_overflow,
            prior_spr2,
            spr2,
        })
    }
}
