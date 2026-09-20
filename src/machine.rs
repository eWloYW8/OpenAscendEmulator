use crate::architecture::Architecture;
use crate::isa::{
    AicDecoderHint, ScalarKey0Operation, ScalarKey7Operation, ScalarLoadStoreOperation,
};
use crate::scalar::{ScalarIntegerError, evaluate_scalar_integer_immediate, update_overflow_spr2};
use serde::Serialize;
use thiserror::Error;

pub const SCALAR_X_REGISTER_COUNT: usize = 32;
const SCALAR_SPR_SNAPSHOT_CAPACITY: usize = 243;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScalarMachine {
    architecture: Architecture,
    xregs: [u64; SCALAR_X_REGISTER_COUNT],
    spr2: u64,
    spr_values: [Option<u64>; SCALAR_SPR_SNAPSHOT_CAPACITY],
    model_time: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarStep {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub source_register: Option<u8>,
    pub source_value: Option<u64>,
    pub second_source_register: Option<u8>,
    pub second_source_value: Option<u64>,
    pub value: u64,
    pub signed_overflow: bool,
    pub prior_spr2: u64,
    pub spr2: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSprStep {
    pub pc: u64,
    pub word: u32,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
    pub source_register: u8,
    pub source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarSprReadSource {
    RegisterSnapshot,
    ProgramCounter,
    ModelTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarSprReadStep {
    pub pc: u64,
    pub word: u32,
    pub destination_register: u8,
    pub prior_destination_value: u64,
    pub source_spr: u16,
    pub source: ScalarSprReadSource,
    pub source_value: u64,
    pub value: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ScalarInstructionStep {
    Register(ScalarStep),
    SprRead(ScalarSprReadStep),
    SprWrite(ScalarSprStep),
    Memory(ScalarMemoryStep),
}

pub trait ScalarMemoryBus {
    type Error: std::error::Error + 'static;

    fn read(&mut self, effective_address: u64, destination: &mut [u8]) -> Result<(), Self::Error>;
    fn write(&mut self, effective_address: u64, source: &[u8]) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ScalarMemoryStep {
    pub pc: u64,
    pub word: u32,
    pub operation: ScalarLoadStoreOperation,
    pub effective_address: u64,
    pub width_bytes: u8,
    pub data_register: u8,
    pub prior_data_value: u64,
    pub data_value: u64,
    pub base_register: u8,
    pub prior_base_value: u64,
    pub updated_base: Option<u64>,
    pub bytes: [u8; 8],
    pub sign_extension_requested: bool,
}

#[derive(Debug, Error)]
pub enum ScalarMemoryExecutionError<E: std::error::Error + 'static> {
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented scalar LD/ST instruction")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error(
        "post-indexed word {word:#010x} at PC {pc:#x} aliases data and base register X{register}"
    )]
    AliasedPostIndex { pc: u64, word: u32, register: u8 },
    #[error("scalar memory backend failed: {0}")]
    Backend(#[source] E),
}

#[derive(Debug, Error)]
pub enum ScalarInstructionError<E: std::error::Error + 'static> {
    #[error(transparent)]
    Scalar(#[from] ScalarMachineError),
    #[error(transparent)]
    Memory(#[from] ScalarMemoryExecutionError<E>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScalarMachineError {
    #[error("X-register index {0} is outside X0..X31")]
    RegisterOutOfRange(u8),
    #[error("SPR index {0} is outside the modeled architecture's register range")]
    SprRegisterOutOfRange(u16),
    #[error("SPR {spr} has no supplied value at PC {pc:#x}")]
    SprValueUnavailable { pc: u64, spr: u16 },
    #[error("model time has no supplied value at PC {pc:#x}")]
    ModelTimeUnavailable { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented scalar instruction")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error(transparent)]
    Arithmetic(#[from] ScalarIntegerError),
}

impl ScalarMachine {
    pub const fn new(
        architecture: Architecture,
        xregs: [u64; SCALAR_X_REGISTER_COUNT],
        spr2: u64,
    ) -> Self {
        Self {
            architecture,
            xregs,
            spr2,
            spr_values: [None; SCALAR_SPR_SNAPSHOT_CAPACITY],
            model_time: None,
        }
    }

    pub const fn architecture(&self) -> Architecture {
        self.architecture
    }

    pub const fn xregs(&self) -> &[u64; SCALAR_X_REGISTER_COUNT] {
        &self.xregs
    }

    pub const fn spr2(&self) -> u64 {
        self.spr2
    }

    pub fn spr_value(&self, index: u16) -> Option<u64> {
        self.spr_values.get(usize::from(index)).copied().flatten()
    }

    pub const fn model_time(&self) -> Option<u64> {
        self.model_time
    }

    pub fn set_model_time(&mut self, ticks: u64) {
        self.model_time = Some(ticks);
    }

    pub fn set_spr_value(&mut self, index: u16, value: u64) -> Result<(), ScalarMachineError> {
        let limit = match self.architecture {
            Architecture::Dav2201 => 187,
            Architecture::Dav3510 => SCALAR_SPR_SNAPSHOT_CAPACITY,
        };
        let slot = self
            .spr_values
            .get_mut(usize::from(index))
            .filter(|_| usize::from(index) < limit)
            .ok_or(ScalarMachineError::SprRegisterOutOfRange(index))?;
        *slot = Some(value);
        Ok(())
    }

    pub fn execute_spr_read_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarSprReadStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarKey2MoveFromSpr {
            destination_register,
            encoded_source_spr,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let limit = match self.architecture {
            Architecture::Dav2201 => 187,
            Architecture::Dav3510 => SCALAR_SPR_SNAPSHOT_CAPACITY,
        };
        if usize::from(encoded_source_spr) >= limit {
            return Err(ScalarMachineError::SprRegisterOutOfRange(
                encoded_source_spr,
            ));
        }
        let (source, source_value) = match (self.architecture, encoded_source_spr) {
            (_, 0) => (ScalarSprReadSource::ProgramCounter, pc),
            (Architecture::Dav3510, 72) => (
                ScalarSprReadSource::ModelTime,
                self.model_time
                    .ok_or(ScalarMachineError::ModelTimeUnavailable { pc })?,
            ),
            _ => (
                ScalarSprReadSource::RegisterSnapshot,
                self.spr_values[usize::from(encoded_source_spr)].ok_or(
                    ScalarMachineError::SprValueUnavailable {
                        pc,
                        spr: encoded_source_spr,
                    },
                )?,
            ),
        };
        let value = if self.architecture == Architecture::Dav2201 && encoded_source_spr == 7 {
            source_value & !1
        } else {
            source_value
        };
        let destination = &mut self.xregs[usize::from(destination_register)];
        let prior_destination_value = *destination;
        *destination = value;
        Ok(ScalarSprReadStep {
            pc,
            word,
            destination_register,
            prior_destination_value,
            source_spr: encoded_source_spr,
            source,
            source_value,
            value,
        })
    }

    pub fn execute_spr_word(
        &mut self,
        pc: u64,
        word: u32,
    ) -> Result<ScalarSprStep, ScalarMachineError> {
        let Some(AicDecoderHint::ScalarKey2MoveToSpr {
            encoded_destination_spr,
            source_register,
            ..
        }) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let mask = match (self.architecture, encoded_destination_spr) {
            (Architecture::Dav2201, 3) | (Architecture::Dav3510, 3 | 105 | 112) => u64::MAX,
            (Architecture::Dav3510, 90) => 0xff,
            _ => return Err(ScalarMachineError::UnsupportedWord { pc, word }),
        };
        let Some(&source_value) = self.xregs.get(usize::from(source_register)) else {
            return Err(ScalarMachineError::UnsupportedWord { pc, word });
        };
        let slot = &mut self.spr_values[usize::from(encoded_destination_spr)];
        let prior_destination_value = *slot;
        let value = source_value & mask;
        *slot = Some(value);
        Ok(ScalarSprStep {
            pc,
            word,
            destination_spr: encoded_destination_spr,
            prior_destination_value,
            source_register,
            source_value,
            value,
        })
    }

    pub fn set_xreg(&mut self, register: u8, value: u64) -> Result<(), ScalarMachineError> {
        let slot = self
            .xregs
            .get_mut(usize::from(register))
            .ok_or(ScalarMachineError::RegisterOutOfRange(register))?;
        *slot = value;
        Ok(())
    }

    pub fn execute_word(&mut self, pc: u64, word: u32) -> Result<ScalarStep, ScalarMachineError> {
        let hint = AicDecoderHint::from_word(self.architecture, word)
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
            AicDecoderHint::ScalarKey0 {
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
                    (ScalarKey0Operation::And, 3) => {
                        (first_source_value & second_source_value, false)
                    }
                    (ScalarKey0Operation::Or, 3) => {
                        (first_source_value | second_source_value, false)
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
            AicDecoderHint::ScalarKey2ShiftLeft {
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
            AicDecoderHint::ScalarKey2MoveRegister {
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
            AicDecoderHint::ScalarKey2ZeroExtend {
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
            AicDecoderHint::ScalarKey7 {
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
            AicDecoderHint::ScalarKey8 {
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

    pub fn execute_instruction<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarInstructionStep, ScalarInstructionError<B::Error>> {
        match AicDecoderHint::from_word(self.architecture, word) {
            Some(AicDecoderHint::ScalarLoadStoreImmediate { .. }) => Ok(
                ScalarInstructionStep::Memory(self.execute_memory_word(pc, word, bus)?),
            ),
            Some(AicDecoderHint::ScalarKey2MoveFromSpr { .. }) => Ok(
                ScalarInstructionStep::SprRead(self.execute_spr_read_word(pc, word)?),
            ),
            Some(AicDecoderHint::ScalarKey2MoveToSpr { .. }) => Ok(
                ScalarInstructionStep::SprWrite(self.execute_spr_word(pc, word)?),
            ),
            _ => Ok(ScalarInstructionStep::Register(
                self.execute_word(pc, word)?,
            )),
        }
    }

    pub fn execute_memory_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarMemoryStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(
            hint @ AicDecoderHint::ScalarLoadStoreImmediate {
                operation,
                width_bytes,
                data_register,
                base_register,
                post_index,
                sign_extend,
                ..
            },
        ) = AicDecoderHint::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        if post_index
            && matches!(operation, ScalarLoadStoreOperation::Load)
            && data_register == base_register
        {
            return Err(ScalarMemoryExecutionError::AliasedPostIndex {
                pc,
                word,
                register: data_register,
            });
        }
        let prior_base_value = self.xregs[usize::from(base_register)];
        let prior_data_value = self.xregs[usize::from(data_register)];
        let effect = hint
            .scalar_address_effect(prior_base_value)
            .expect("the load/store hint has an address effect");
        let width = usize::from(width_bytes);
        let mut bytes = [0_u8; 8];
        let sign_extension_requested = sign_extend == Some(true) && width_bytes < 8;
        let data_value = match operation {
            ScalarLoadStoreOperation::Load => {
                bus.read(effect.effective_address, &mut bytes[..width])
                    .map_err(ScalarMemoryExecutionError::Backend)?;
                let raw = u64::from_le_bytes(bytes);
                if sign_extension_requested {
                    let bits = u32::from(width_bytes) * 8;
                    (((raw << (64 - bits)) as i64) >> (64 - bits)) as u64
                } else {
                    raw
                }
            }
            ScalarLoadStoreOperation::Store => {
                bytes[..width].copy_from_slice(&prior_data_value.to_le_bytes()[..width]);
                bus.write(effect.effective_address, &bytes[..width])
                    .map_err(ScalarMemoryExecutionError::Backend)?;
                prior_data_value
            }
        };
        if matches!(operation, ScalarLoadStoreOperation::Load) {
            self.xregs[usize::from(data_register)] = data_value;
        }
        if let Some(updated_base) = effect.updated_base {
            self.xregs[usize::from(base_register)] = updated_base;
        }
        Ok(ScalarMemoryStep {
            pc,
            word,
            operation,
            effective_address: effect.effective_address,
            width_bytes,
            data_register,
            prior_data_value,
            data_value,
            base_register,
            prior_base_value,
            updated_base: effect.updated_base,
            bytes,
            sign_extension_requested,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBus {
        base: u64,
        bytes: [u8; 32],
        fail: bool,
        accesses: usize,
    }

    impl TestBus {
        fn new(base: u64) -> Self {
            Self {
                base,
                bytes: [0; 32],
                fail: false,
                accesses: 0,
            }
        }

        fn range(&self, address: u64, len: usize) -> std::io::Result<std::ops::Range<usize>> {
            let start = usize::try_from(
                address
                    .checked_sub(self.base)
                    .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidInput))?,
            )
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
            let end = start
                .checked_add(len)
                .filter(|end| *end <= self.bytes.len())
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
            Ok(start..end)
        }
    }

    impl ScalarMemoryBus for TestBus {
        type Error = std::io::Error;

        fn read(&mut self, address: u64, destination: &mut [u8]) -> std::io::Result<()> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            let range = self.range(address, destination.len())?;
            destination.copy_from_slice(&self.bytes[range]);
            self.accesses += 1;
            Ok(())
        }

        fn write(&mut self, address: u64, source: &[u8]) -> std::io::Result<()> {
            if self.fail {
                return Err(std::io::Error::from(std::io::ErrorKind::BrokenPipe));
            }
            let range = self.range(address, source.len())?;
            self.bytes[range].copy_from_slice(source);
            self.accesses += 1;
            Ok(())
        }
    }

    #[test]
    fn unified_scalar_dispatch_preserves_spr_memory_dependency_on_both_architectures() {
        for (architecture, load_word) in [
            (Architecture::Dav2201, 0x03c2_0008),
            (Architecture::Dav3510, 0x1cc2_0008),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_spr_value(4, 0x1000).unwrap();
            let mut bus = TestBus::new(0x1000);
            let expected = 0x1234_5678_9abc_def0_u64;
            bus.bytes[8..16].copy_from_slice(&expected.to_le_bytes());

            let read = machine
                .execute_instruction(0x200, 0x0200_4880, &mut bus)
                .unwrap();
            let ScalarInstructionStep::SprRead(read) = read else {
                panic!("expected SPR read");
            };
            assert_eq!(read.value, 0x1000);
            assert_eq!(machine.xregs()[0], 0x1000);

            bus.fail = true;
            let before = machine.clone();
            assert!(matches!(
                machine.execute_instruction(0x204, load_word, &mut bus),
                Err(ScalarInstructionError::Memory(
                    ScalarMemoryExecutionError::Backend(_)
                ))
            ));
            assert_eq!(machine, before);
            bus.fail = false;

            let load = machine
                .execute_instruction(0x204, load_word, &mut bus)
                .unwrap();
            let ScalarInstructionStep::Memory(load) = load else {
                panic!("expected memory load");
            };
            assert_eq!(load.effective_address, 0x1008);
            assert_eq!(load.data_value, expected);
            assert_eq!(machine.xregs()[1], expected);
            assert_eq!(bus.accesses, 1);

            let write = machine
                .execute_instruction(0x208, 0x0206_1900, &mut bus)
                .unwrap();
            let ScalarInstructionStep::SprWrite(write) = write else {
                panic!("expected SPR write");
            };
            assert_eq!(write.destination_spr, 3);
            assert_eq!(write.value, expected);
            assert_eq!(machine.spr_value(3), Some(expected));
            assert_eq!(machine.spr2(), 0x55);

            let register = machine
                .execute_instruction(0x20c, 0x0706_0001, &mut bus)
                .unwrap();
            assert!(matches!(register, ScalarInstructionStep::Register(_)));
            assert_eq!(machine.xregs()[3], 1);
            let before = machine.clone();
            assert!(matches!(
                machine.execute_instruction(0x210, 0x6000_0000, &mut bus),
                Err(ScalarInstructionError::Scalar(
                    ScalarMachineError::UnsupportedWord { .. }
                ))
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn c220_narrow_load_zero_extends_and_negative_offset_selects_access_address() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(5, 0x1000).unwrap();
        machine.set_xreg(7, u64::MAX).unwrap();
        let mut bus = TestBus::new(0x1000 - 584);
        bus.bytes[0] = 0x80;
        let word = 0x03ce_5db8 & !(3 << 22);
        let step = machine.execute_memory_word(0x100, word, &mut bus).unwrap();
        assert_eq!(step.operation, ScalarLoadStoreOperation::Load);
        assert_eq!(step.effective_address, 0x1000 - 584);
        assert_eq!(step.width_bytes, 1);
        assert_eq!(step.prior_data_value, u64::MAX);
        assert_eq!(step.data_value, 0x80);
        assert_eq!(machine.xregs()[7], 0x80);
        assert_eq!(machine.xregs()[5], 0x1000);
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn c310_signed_post_index_load_sign_extends_and_updates_base() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(5, 0x1000).unwrap();
        let mut bus = TestBus::new(0x1000);
        bus.bytes[0] = 0x80;
        let word = 0x1fce_5db8 & !(3 << 22);
        let step = machine.execute_memory_word(0x104, word, &mut bus).unwrap();
        assert_eq!(step.effective_address, 0x1000);
        assert_eq!(step.updated_base, Some(0x1000 - 584));
        assert!(step.sign_extension_requested);
        assert_eq!(step.data_value, (-128_i64) as u64);
        assert_eq!(machine.xregs()[7], (-128_i64) as u64);
        assert_eq!(machine.xregs()[5], 0x1000 - 584);
    }

    #[test]
    fn c310_sign_extension_applies_to_one_two_and_four_byte_loads() {
        let mut bus = TestBus::new(0x1000);
        bus.bytes[..8].copy_from_slice(&[0x78, 0x80, 0x34, 0x80, 0, 0, 0, 0x80]);
        for (dtype_bits, expected) in [
            (0_u32, 0x78_u64),
            (1_u32, 0xffff_ffff_ffff_8078_u64),
            (2_u32, 0xffff_ffff_8034_8078_u64),
            (3_u32, 0x8000_0000_8034_8078_u64),
        ] {
            let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
            machine.set_xreg(5, 0x1000).unwrap();
            let word = (0x1fce_5db8 & !(3 << 22)) | (dtype_bits << 22);
            let step = machine.execute_memory_word(0, word, &mut bus).unwrap();
            assert_eq!(step.data_value, expected);
            assert_eq!(machine.xregs()[7], expected);
        }
    }

    #[test]
    fn both_architectures_store_low_bytes_in_little_endian_order() {
        for (architecture, word) in [
            (Architecture::Dav2201, 0x04c6_6000),
            (Architecture::Dav3510, 0x03c6_6000),
        ] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(6, 0x2000).unwrap();
            machine.set_xreg(3, 0x0123_4567_89ab_cdef).unwrap();
            let mut bus = TestBus::new(0x2000);
            let step = machine.execute_memory_word(0x108, word, &mut bus).unwrap();
            assert_eq!(step.operation, ScalarLoadStoreOperation::Store);
            assert_eq!(step.width_bytes, 8);
            assert_eq!(step.bytes, [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01]);
            assert_eq!(bus.bytes[..8], step.bytes);
            assert_eq!(machine.xregs()[3], 0x0123_4567_89ab_cdef);
        }
    }

    #[test]
    fn c310_post_index_store_can_alias_its_data_register() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        machine.set_xreg(5, 0x3000).unwrap();
        let mut bus = TestBus::new(0x3000);
        let aliased = (0x13ce_5db8 & !(0x1f << 17)) | (5 << 17);
        let step = machine
            .execute_memory_word(0x10c, aliased, &mut bus)
            .unwrap();
        assert_eq!(step.data_value, 0x3000);
        assert_eq!(step.effective_address, 0x3000);
        assert_eq!(bus.bytes[..8], 0x3000_u64.to_le_bytes());
        assert_eq!(machine.xregs()[5], 0x3000 - 584);
    }

    #[test]
    fn memory_backend_error_and_aliased_post_index_do_not_change_registers() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [7; 32], 0x55);
        let before = machine.clone();
        let mut bus = TestBus::new(7);
        bus.fail = true;
        assert!(matches!(
            machine.execute_memory_word(0x100, 0x1fce_5db8, &mut bus),
            Err(ScalarMemoryExecutionError::Backend(_))
        ));
        assert_eq!(machine, before);
        assert_eq!(bus.accesses, 0);
        let aliased = (0x1fce_5db8 & !(0x1f << 17)) | (5 << 17);
        assert!(matches!(
            machine.execute_memory_word(0x100, aliased, &mut bus),
            Err(ScalarMemoryExecutionError::AliasedPostIndex { register: 5, .. })
        ));
        assert_eq!(machine, before);
        assert_eq!(bus.accesses, 0);
    }

    #[test]
    fn observed_greater_move_and_arithmetic_chain_matches_register_values() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            assert_eq!(
                machine.execute_word(0x112f5000, 0x073a7f80).unwrap().value,
                0x7f80
            );
            assert_eq!(
                machine.execute_word(0x112f5004, 0x077b0010).unwrap().value,
                0x107f80
            );
            machine.set_xreg(29, 0x1c7f80).unwrap();
            assert_eq!(
                machine.execute_word(0x112f5034, 0x083dd0a8).unwrap().value,
                0x1c8028
            );
            let step = machine.execute_word(0x112f503c, 0x0893e790).unwrap();
            assert_eq!(step.source_register, Some(30));
            assert_eq!(step.source_value, Some(0x1c8028));
            assert_eq!(step.value, 0x1c7898);
            assert_eq!(machine.xregs()[9], 0x1c7898);

            assert_eq!(
                machine.execute_word(0x112f5138, 0x070c_a000).unwrap().value,
                0xa000
            );
            assert_eq!(
                machine.execute_word(0x112f513c, 0x078d_ffff).unwrap().value,
                0xffff_0000_a000
            );
            assert_eq!(
                machine.execute_word(0x112f5140, 0x07cd_ffff).unwrap().value,
                0xffff_ffff_0000_a000
            );
        }
    }

    #[test]
    fn movk_preserves_three_other_halfwords_and_supports_all_lanes() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(6, 0x0123_4567_89ab_cdef).unwrap();
        for (word, expected) in [
            (0x070d_ffff, 0x0123_4567_89ab_ffff),
            (0x074d_0001, 0x0123_4567_0001_ffff),
            (0x078d_0002, 0x0123_0002_0001_ffff),
            (0x07cd_0003, 0x0003_0002_0001_ffff),
        ] {
            assert_eq!(machine.execute_word(0, word).unwrap().value, expected);
            assert_eq!(machine.spr2(), 0x55);
        }
    }

    #[test]
    fn overflow_updates_spr2_and_unsupported_words_do_not_mutate_state() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x1234);
        machine.set_xreg(1, i64::MAX as u64).unwrap();
        let step = machine.execute_word(0x100, 0x0802_1001).unwrap();
        assert!(step.signed_overflow);
        assert_eq!(machine.xregs()[1], i64::MIN as u64);
        assert_eq!(machine.spr2(), 0x401234);
        let snapshot = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x08c0_0000),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, snapshot);
        assert_eq!(
            machine.set_xreg(32, 0),
            Err(ScalarMachineError::RegisterOutOfRange(32))
        );
    }

    #[test]
    fn decoded_memory_words_do_not_mutate_the_register_only_machine() {
        for (architecture, words) in [
            (Architecture::Dav2201, [0x03ce_5db8, 0x04c6_6000]),
            (Architecture::Dav3510, [0x1cce_5db8, 0x03ce_5db8]),
        ] {
            let mut machine = ScalarMachine::new(architecture, [7; 32], 0x55);
            let snapshot = machine.clone();
            for word in words {
                assert!(matches!(
                    machine.execute_word(0x100, word),
                    Err(ScalarMachineError::UnsupportedWord { .. })
                ));
                assert_eq!(machine, snapshot);
            }
        }
    }

    #[test]
    fn zero_extend_masks_valid_widths_and_rejects_unmodeled_high_registers() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(3, 0x1234_5678_9abc_def0).unwrap();
            for (word, value) in [
                (0x0208_3a00, 0xf0),
                (0x0248_3a00, 0xdef0),
                (0x0288_3a00, 0x9abc_def0),
            ] {
                let step = machine.execute_word(0x100, word).unwrap();
                assert_eq!(step.source_register, Some(3));
                assert_eq!(step.source_value, Some(0x1234_5678_9abc_def0));
                assert_eq!(step.value, value);
                assert_eq!(machine.xregs()[4], value);
                assert_eq!(machine.spr2(), 0x55);
            }
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x0208_3a60),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn register_move_copies_all_bits_without_modifying_spr2() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(30, 0xfedc_ba98_7654_3210).unwrap();
            let step = machine.execute_word(0x100, 0x0209_e800).unwrap();
            assert_eq!(step.source_register, Some(30));
            assert_eq!(step.source_value, Some(0xfedc_ba98_7654_3210));
            assert_eq!(step.value, 0xfedc_ba98_7654_3210);
            assert_eq!(machine.xregs()[4], 0xfedc_ba98_7654_3210);
            assert_eq!(machine.spr2(), 0x55);
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x104, 0x0209_e860),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn register_binary_forms_use_two_sources_and_update_only_verified_state() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x55);
            machine.set_xreg(29, 0x0010_7fa0).unwrap();
            machine.set_xreg(15, 0).unwrap();
            let add = machine.execute_word(0x126e_cb04, 0x003b_d781).unwrap();
            assert_eq!(add.value, 0x0010_7fa0);
            assert_eq!(add.second_source_register, Some(15));
            assert_eq!(add.second_source_value, Some(0));
            machine.set_xreg(8, 0x209).unwrap();
            machine.set_xreg(7, 1).unwrap();
            let mul = machine.execute_word(0x126e_d004, 0x000e_8383).unwrap();
            assert_eq!(mul.value, 0x209);
            assert_eq!(machine.xregs()[7], 0x209);
            machine.set_xreg(15, 0x18).unwrap();
            machine.set_xreg(16, 0x7fff).unwrap();
            assert_eq!(
                machine
                    .execute_word(0x126e_cb14, 0x00de_f80a)
                    .unwrap()
                    .value,
                0x18
            );
            machine.set_xreg(1, 0xc000_0000_0000_0000).unwrap();
            machine.set_xreg(0, 0x0808_0801_0101).unwrap();
            let or = machine.execute_word(0x126e_d29c, 0x00c0_100b).unwrap();
            assert_eq!(or.value, 0xc000_0808_0801_0101);
            assert_eq!(or.prior_destination_value, 0x0808_0801_0101);
            assert_eq!(or.second_source_value, Some(0x0808_0801_0101));
            assert_eq!(machine.spr2(), 0x55);
        }
    }

    #[test]
    fn register_add_and_multiply_overflow_update_spr2_and_reject_unmodeled_forms() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            machine.set_xreg(1, i64::MAX as u64).unwrap();
            machine.set_xreg(2, 1).unwrap();
            let add = machine.execute_word(0x100, 0x0002_1101).unwrap();
            assert_eq!(add.value, i64::MIN as u64);
            assert!(add.signed_overflow);
            assert_eq!(machine.spr2(), 0x400010);
            machine.set_xreg(2, u64::MAX).unwrap();
            let mul = machine.execute_word(0x104, 0x0002_1103).unwrap();
            assert_eq!(mul.value, i64::MIN as u64);
            assert!(mul.signed_overflow);
            assert_eq!(machine.spr2(), 0x410010);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x0082_1101),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        let before = machine.clone();
        assert!(matches!(
            machine.execute_word(0x100, 0x003b_d7f1),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(machine, before);
    }

    #[test]
    fn multiply_add_uses_prior_destination_and_preserves_overflow_state() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x24);
            machine.set_xreg(29, 0x107f80).unwrap();
            machine.set_xreg(15, 0x18).unwrap();
            machine.set_xreg(17, 0x8000).unwrap();
            let step = machine.execute_word(0x112f5028, 0x003a_f884).unwrap();
            assert_eq!(step.prior_destination_value, 0x107f80);
            assert_eq!(step.source_register, Some(15));
            assert_eq!(step.second_source_register, Some(17));
            assert_eq!(step.value, 0x1c7f80);
            assert!(!step.signed_overflow);
            assert_eq!(step.spr2, 0x24);

            machine.set_xreg(29, 0).unwrap();
            machine.set_xreg(15, i64::MIN as u64).unwrap();
            machine.set_xreg(17, u64::MAX).unwrap();
            let product_overflow = machine.execute_word(0x100, 0x003a_f884).unwrap();
            assert_eq!(product_overflow.value, i64::MIN as u64);
            assert!(product_overflow.signed_overflow);
            assert_eq!(product_overflow.spr2, 0x400034);

            machine.set_xreg(29, i64::MAX as u64).unwrap();
            machine.set_xreg(15, 1).unwrap();
            machine.set_xreg(17, 1).unwrap();
            let add_overflow = machine.execute_word(0x104, 0x003a_f884).unwrap();
            assert_eq!(add_overflow.value, i64::MIN as u64);
            assert!(add_overflow.signed_overflow);
            assert_eq!(add_overflow.spr2, 0x410034);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x00ba_f884),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn shift_left_uses_old_destination_and_masks_register_count() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0x1234);
            machine.set_xreg(4, 0x38).unwrap();
            let immediate = machine.execute_word(0x100, 0x02c8_020f).unwrap();
            assert_eq!(immediate.value, 0x1c_0000);
            assert_eq!(immediate.source_register, Some(4));
            assert_eq!(immediate.source_value, Some(0x38));
            assert_eq!(immediate.second_source_register, None);

            machine.set_xreg(10, 0xffff_ffff).unwrap();
            machine.set_xreg(22, 0x60).unwrap();
            let register = machine.execute_word(0x104, 0x02d5_6240).unwrap();
            assert_eq!(register.value, 0xffff_ffff_0000_0000);
            assert_eq!(register.source_register, Some(10));
            assert_eq!(register.second_source_register, Some(22));
            assert_eq!(register.second_source_value, Some(0x60));
            assert_eq!(machine.spr2(), 0x1234);

            let before = machine.clone();
            assert!(matches!(
                machine.execute_word(0x108, 0x0288_020f),
                Err(ScalarMachineError::UnsupportedWord { .. })
            ));
            assert_eq!(machine, before);
        }
    }

    #[test]
    fn observed_c220_mov_spr_xn_writes_full_width_ctrl_value() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_xreg(19, 0x0100_0000_0000_0000).unwrap();
        let first = machine.execute_spr_word(0x1131_2644, 0x0207_3900).unwrap();
        assert_eq!(first.destination_spr, 3);
        assert_eq!(first.source_register, 19);
        assert_eq!(first.source_value, 0x0100_0000_0000_0000);
        assert_eq!(first.prior_destination_value, None);
        assert_eq!(machine.spr_value(3), Some(0x0100_0000_0000_0000));

        machine.set_xreg(19, 0).unwrap();
        let second = machine.execute_spr_word(0x1131_2654, 0x0207_3900).unwrap();
        assert_eq!(second.prior_destination_value, Some(first.value));
        assert_eq!(machine.spr_value(3), Some(0));
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn observed_c310_mov_spr_xn_applies_spr90_mask() {
        let mut machine = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x55);
        machine.set_xreg(0, 0x1234_5678_9abc_def0).unwrap();
        for (word, destination, expected) in [
            (0x0206_0900, 3, 0x1234_5678_9abc_def0),
            (0x02b4_0900, 90, 0xf0),
            (0x02d2_0900, 105, 0x1234_5678_9abc_def0),
            (0x02e0_0900, 112, 0x1234_5678_9abc_def0),
        ] {
            let step = machine.execute_spr_word(0x10d0_d140, word).unwrap();
            assert_eq!(step.destination_spr, destination);
            assert_eq!(step.value, expected);
            assert_eq!(machine.spr_value(destination), Some(expected));
        }
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn c220_spr_reads_feed_the_captured_scalar_prefix() {
        let mut machine = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        machine.set_spr_value(67, 0).unwrap();
        machine.set_spr_value(16, 0).unwrap();
        machine.set_spr_value(4, 0x1022_be00).unwrap();
        machine.execute_word(0x10d0_d000, 0x073a_7fa0).unwrap();
        machine.execute_word(0x10d0_d004, 0x077b_0010).unwrap();
        let read = machine
            .execute_spr_read_word(0x10d0_d008, 0x029e_3880)
            .unwrap();
        assert_eq!(read.source_spr, 67);
        assert_eq!(read.source, ScalarSprReadSource::RegisterSnapshot);
        assert_eq!(read.destination_register, 15);
        assert_eq!(read.value, 0);
        assert_eq!(machine.xregs()[15], 0);
        assert_eq!(
            machine
                .execute_word(0x10d0_d00c, 0x003b_d781)
                .unwrap()
                .value,
            0x107fa0
        );
        assert_eq!(
            machine
                .execute_spr_read_word(0x10d0_d010, 0x021f_0880)
                .unwrap()
                .source_spr,
            16
        );
        let parameter_base = machine
            .execute_spr_read_word(0x10d0_d028, 0x0200_4880)
            .unwrap();
        assert_eq!(parameter_base.value, 0x1022_be00);
        assert_eq!(machine.xregs()[0], 0x1022_be00);
        assert_eq!(machine.spr2(), 0x55);
    }

    #[test]
    fn spr_read_special_sources_and_missing_values_are_explicit() {
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut machine = ScalarMachine::new(architecture, [0; 32], 0);
            let pc = 0x10d0_d000;
            let read = machine.execute_spr_read_word(pc, 0x0200_0880).unwrap();
            assert_eq!(read.source_spr, 0);
            assert_eq!(read.source, ScalarSprReadSource::ProgramCounter);
            assert_eq!(read.value, pc);
            assert_eq!(machine.xregs()[0], pc);

            let before = machine.clone();
            assert_eq!(
                machine.execute_spr_read_word(pc + 4, 0x0200_4880),
                Err(ScalarMachineError::SprValueUnavailable { pc: pc + 4, spr: 4 })
            );
            assert_eq!(machine, before);
            machine.set_spr_value(7, 0x1235).unwrap();
            let read = machine.execute_spr_read_word(pc + 8, 0x0200_7880).unwrap();
            assert_eq!(read.source_value, 0x1235);
            assert_eq!(
                read.value,
                if architecture == Architecture::Dav2201 {
                    0x1234
                } else {
                    0x1235
                }
            );
            assert_eq!(machine.spr2(), 0);
        }

        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0);
        let pc = 0x100;
        assert_eq!(
            c310.execute_spr_read_word(pc, 0x0280_8880),
            Err(ScalarMachineError::ModelTimeUnavailable { pc })
        );
        c310.set_model_time(0x5678);
        let read = c310.execute_spr_read_word(pc, 0x0280_8880).unwrap();
        assert_eq!(read.source_spr, 72);
        assert_eq!(read.source, ScalarSprReadSource::ModelTime);
        assert_eq!(read.value, 0x5678);
        assert_eq!(c310.xregs()[0], 0x5678);
        assert_eq!(c310.model_time(), Some(0x5678));
        assert_eq!(
            c310.set_spr_value(243, 1),
            Err(ScalarMachineError::SprRegisterOutOfRange(243))
        );
        let mut c220 = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0);
        assert_eq!(
            c220.set_spr_value(187, 1),
            Err(ScalarMachineError::SprRegisterOutOfRange(187))
        );
    }

    #[test]
    fn unverified_spr_destinations_and_out_of_range_sources_fail_closed() {
        let mut c310 = ScalarMachine::new(Architecture::Dav3510, [0; 32], 0x55);
        let before = c310.clone();
        assert!(matches!(
            c310.execute_spr_word(0x100, 0x0230_0901),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);
        assert!(matches!(
            c310.execute_word(0x100, 0x0206_0900),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c310, before);

        let mut c220 = ScalarMachine::new(Architecture::Dav2201, [0; 32], 0x55);
        let before = c220.clone();
        assert!(matches!(
            c220.execute_spr_word(0x100, 0x0206_1920),
            Err(ScalarMachineError::UnsupportedWord { .. })
        ));
        assert_eq!(c220, before);
    }
}
