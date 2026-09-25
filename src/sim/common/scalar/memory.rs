use super::*;

impl ScalarMachine {
    pub fn execute_memory_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarMemoryStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(
            hint @ ScalarInstruction::ScalarLoadStoreImmediate {
                operation,
                width_bytes,
                data_register,
                base_register,
                post_index,
                sign_extend,
                ..
            },
        ) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        if self.architecture != Architecture::Dav2201
            && post_index
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
        if let Some(updated_base) = effect.updated_base {
            self.xregs[usize::from(base_register)] = updated_base;
        }
        if matches!(operation, ScalarLoadStoreOperation::Load) {
            self.xregs[usize::from(data_register)] = data_value;
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

    pub fn execute_indexed_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarIndexedStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarIndexedStore {
            width_bytes,
            source_register,
            base_register,
            offset_register,
            post_index,
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let value = self.xreg_value(source_register).unwrap_or(0);
        let base_value = self.xreg_value(base_register).unwrap_or(0);
        let offset_value = self.xreg_value(offset_register).unwrap_or(0);
        let indexed_address =
            base_value.wrapping_add(offset_value.wrapping_mul(u64::from(width_bytes)));
        let effective_address = if post_index {
            base_value
        } else {
            indexed_address
        };
        let updated_base = post_index.then_some(indexed_address);
        let bytes = value.to_le_bytes();
        bus.write(effective_address, &bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        if let Some(updated_base) = updated_base {
            self.write_existing_xreg(base_register, updated_base);
        }
        Ok(ScalarIndexedStoreStep {
            pc,
            word,
            effective_address,
            width_bytes,
            source_register,
            value,
            base_register,
            base_value,
            updated_base,
            offset_register,
            offset_value,
            bytes,
        })
    }

    pub fn execute_indexed_load_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarIndexedLoadStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarIndexedLoad {
            width_bytes,
            destination_register,
            base_register,
            offset_register,
            post_index,
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xreg_value(base_register).unwrap_or(0);
        let offset_value = self.xreg_value(offset_register).unwrap_or(0);
        let indexed_address =
            base_value.wrapping_add(offset_value.wrapping_mul(u64::from(width_bytes)));
        let effective_address = if post_index {
            base_value
        } else {
            indexed_address
        };
        let updated_base = post_index.then_some(indexed_address);
        let mut bytes = [0_u8; 8];
        bus.read(effective_address, &mut bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        let value = u64::from_le_bytes(bytes);
        let prior_destination_value = self.xreg_value(destination_register).unwrap_or(0);
        if let Some(updated_base) = updated_base {
            self.write_existing_xreg(base_register, updated_base);
        }
        self.write_existing_xreg(destination_register, value);
        Ok(ScalarIndexedLoadStep {
            pc,
            word,
            effective_address,
            width_bytes,
            destination_register,
            prior_destination_value,
            value,
            base_register,
            base_value,
            updated_base,
            offset_register,
            offset_value,
            bytes,
        })
    }

    pub fn execute_indexed_immediate_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarIndexedImmediateStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarIndexedImmediateStore {
            width_bytes,
            base_register,
            offset_register,
            value,
            post_index,
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let offset_value = self.xregs[usize::from(offset_register)];
        let indexed_address =
            base_value.wrapping_add(offset_value.wrapping_mul(u64::from(width_bytes)));
        let effective_address = if post_index {
            base_value
        } else {
            indexed_address
        };
        let updated_base = post_index.then_some(indexed_address);
        let mut bytes = [0_u8; 8];
        match value {
            ScalarStoreImmediateValue::Zero => {}
            ScalarStoreImmediateValue::One => bytes[0] = 1,
            ScalarStoreImmediateValue::Ones => bytes.fill(0xff),
        }
        bus.write(effective_address, &bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        if let Some(updated_base) = updated_base {
            self.xregs[usize::from(base_register)] = updated_base;
        }
        Ok(ScalarIndexedImmediateStoreStep {
            pc,
            word,
            effective_address,
            width_bytes,
            base_register,
            base_value,
            updated_base,
            offset_register,
            offset_value,
            value,
            bytes,
        })
    }

    pub fn execute_immediate_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarImmediateStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarStoreImmediate {
            width_bytes,
            base_register,
            signed_offset,
            post_index,
            value,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        if post_index && self.architecture != Architecture::Dav2201 {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        }
        let prior_base_value = self.xregs[usize::from(base_register)];
        let adjusted = prior_base_value.wrapping_add(signed_offset as i64 as u64);
        let effective_address = if post_index {
            prior_base_value
        } else {
            adjusted
        };
        let updated_base = post_index.then_some(adjusted);
        let mut bytes = [0_u8; 8];
        match value {
            ScalarStoreImmediateValue::Zero => {}
            ScalarStoreImmediateValue::One => bytes[0] = 1,
            ScalarStoreImmediateValue::Ones => bytes.fill(0xff),
        }
        bus.write(effective_address, &bytes[..usize::from(width_bytes)])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        if let Some(base) = updated_base {
            self.xregs[usize::from(base_register)] = base;
        }
        Ok(ScalarImmediateStoreStep {
            pc,
            word,
            effective_address,
            width_bytes,
            base_register,
            prior_base_value,
            updated_base,
            value,
            bytes,
        })
    }

    pub fn execute_pair_load_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarPairLoadStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarPairLoad {
            width_bytes,
            first_destination_register,
            second_destination_register,
            base_register,
            signed_offset,
            sign_extend,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let first_address = base_value.wrapping_add(signed_offset as i64 as u64);
        let second_address = first_address.wrapping_add(u64::from(width_bytes));
        let width = usize::from(width_bytes);
        let mut first_bytes = [0_u8; 8];
        let mut second_bytes = [0_u8; 8];
        bus.read(first_address, &mut first_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        bus.read(second_address, &mut second_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        let decode = |bytes: [u8; 8]| {
            let raw = u64::from_le_bytes(bytes);
            if sign_extend && width_bytes < 8 {
                let bits = u32::from(width_bytes) * 8;
                (((raw << (64 - bits)) as i64) >> (64 - bits)) as u64
            } else {
                raw
            }
        };
        let first_value = decode(first_bytes);
        let second_value = decode(second_bytes);
        let first_prior_value = self.xregs[usize::from(first_destination_register)];
        let second_prior_value = self.xregs[usize::from(second_destination_register)];
        if self.architecture == Architecture::Dav2201 {
            self.xregs[usize::from(second_destination_register)] = second_value;
            self.xregs[usize::from(first_destination_register)] = first_value;
        } else {
            self.xregs[usize::from(first_destination_register)] = first_value;
            self.xregs[usize::from(second_destination_register)] = second_value;
        }
        Ok(ScalarPairLoadStep {
            pc,
            word,
            first_address,
            second_address,
            width_bytes,
            base_register,
            base_value,
            first_destination_register,
            first_prior_value,
            first_value,
            first_bytes,
            second_destination_register,
            second_prior_value,
            second_value,
            second_bytes,
            sign_extension_requested: sign_extend && width_bytes < 8,
        })
    }

    pub fn execute_pair_store_word<B: ScalarMemoryBus>(
        &mut self,
        pc: u64,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarPairStoreStep, ScalarMemoryExecutionError<B::Error>> {
        let Some(ScalarInstruction::ScalarPairStore {
            width_bytes,
            first_source_register,
            second_source_register,
            base_register,
            signed_offset,
            ..
        }) = ScalarInstruction::from_word(self.architecture, word)
        else {
            return Err(ScalarMemoryExecutionError::UnsupportedWord { pc, word });
        };
        let base_value = self.xregs[usize::from(base_register)];
        let first_address = base_value.wrapping_add(signed_offset as i64 as u64);
        let second_address = first_address.wrapping_add(u64::from(width_bytes));
        let first_value = self.xregs[usize::from(first_source_register)];
        let second_value = self.xregs[usize::from(second_source_register)];
        let width = usize::from(width_bytes);
        let mut first_bytes = [0_u8; 8];
        let mut second_bytes = [0_u8; 8];
        first_bytes[..width].copy_from_slice(&first_value.to_le_bytes()[..width]);
        second_bytes[..width].copy_from_slice(&second_value.to_le_bytes()[..width]);
        bus.write(first_address, &first_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        bus.write(second_address, &second_bytes[..width])
            .map_err(ScalarMemoryExecutionError::Backend)?;
        Ok(ScalarPairStoreStep {
            pc,
            word,
            first_address,
            second_address,
            width_bytes,
            base_register,
            base_value,
            first_source_register,
            first_value,
            first_bytes,
            second_source_register,
            second_value,
            second_bytes,
        })
    }
}
