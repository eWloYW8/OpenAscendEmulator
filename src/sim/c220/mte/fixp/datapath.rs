use super::*;
use crate::sim::c220::mte::interface::C220MteL0cReadInterface;

/// Destination-independent FIX resources. Output layout and command ownership
/// belong to the engine; reading, conversion and dispatch use this one bundle.
#[derive(Debug, Clone)]
pub(super) struct C220FixpDatapath {
    pub read: C220FixpReadPipeline,
    pub input: C220MteL0cReadInterface,
    pub functional: C220FixpFunctionalState,
    pub conversion: C220FixpConversionPipeline,
    pub write: C220FixpDispatchPipeline,
}

impl C220FixpDatapath {
    pub(super) fn new(config: C220FixpEngineConfig) -> Result<Self, C220FixpEngineError> {
        if config.read_bandwidth == 0 {
            return Err(C220FixpReadGeneratorError::ZeroBandwidth.into());
        }
        Ok(Self {
            read: C220FixpReadPipeline::default(),
            input: C220MteL0cReadInterface::new(config.read_bank_count, config.read_data_latency)?,
            functional: C220FixpFunctionalState::new(config.l0c_capacity),
            conversion: C220FixpConversionPipeline::default(),
            write: C220FixpDispatchPipeline::default(),
        })
    }

    pub(super) fn is_idle(&self) -> bool {
        self.read.is_idle()
            && self.input.is_idle()
            && self.conversion.entries().is_empty()
            && self.write.is_idle()
    }

    pub(super) fn admission_backpressure(
        &self,
        instructions: usize,
        capacity: u32,
    ) -> Option<C220FixpAdmission> {
        if self.read.generated_batches() != 0 {
            Some(C220FixpAdmission::ReadGenerationBusy)
        } else if instructions >= capacity as usize {
            Some(C220FixpAdmission::InstructionFifoFull)
        } else {
            None
        }
    }
}
