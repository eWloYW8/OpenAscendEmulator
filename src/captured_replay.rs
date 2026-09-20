use serde::Serialize;
use thiserror::Error;

use crate::{
    Architecture, C220CapturedMaskedAddError, C220CapturedMaskedAddRun, C220CapturedMulError,
    C220CapturedMulRun, C220CapturedSubError, C220CapturedSubRun, C310CapturedMaskedAddError,
    C310CapturedMaskedAddRun, C310CapturedMulError, C310CapturedMulRun, C310CapturedSubError,
    C310CapturedSubRun, execute_captured_c220_masked_add, execute_captured_c220_mul,
    execute_captured_c220_mul_predecessor_chains, execute_captured_c220_sub,
    execute_captured_c220_sub_predecessor_chains, execute_captured_c310_masked_add,
    execute_captured_c310_mul, execute_captured_c310_mul_predecessor_chains,
    execute_captured_c310_sub, execute_captured_c310_sub_predecessor_chains,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapturedReplayOperation {
    MaskedAdd,
    Subtract,
    Multiply,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "template", content = "run", rename_all = "snake_case")]
pub enum CapturedReplay {
    C220MaskedAdd(C220CapturedMaskedAddRun),
    C310MaskedAdd(C310CapturedMaskedAddRun),
    C220Subtract(C220CapturedSubRun),
    C310Subtract(C310CapturedSubRun),
    C220Multiply(C220CapturedMulRun),
    C310Multiply(C310CapturedMulRun),
}

impl CapturedReplay {
    pub fn output(&self) -> &[u8] {
        match self {
            Self::C220MaskedAdd(run) => &run.output,
            Self::C310MaskedAdd(run) => &run.output,
            Self::C220Subtract(run) => &run.output,
            Self::C310Subtract(run) => &run.output,
            Self::C220Multiply(run) => &run.output,
            Self::C310Multiply(run) => &run.output,
        }
    }

    pub fn tile_count(&self) -> usize {
        match self {
            Self::C220MaskedAdd(run) => run.tiles.len(),
            Self::C310MaskedAdd(run) => run.tiles.len(),
            Self::C220Subtract(run) => run.tiles.len(),
            Self::C310Subtract(run) => run.tiles.len(),
            Self::C220Multiply(run) => run.tiles.len(),
            Self::C310Multiply(run) => run.tiles.len(),
        }
    }
}

#[derive(Debug, Error)]
pub enum CapturedReplayError {
    #[error("masked Add requires an explicit prior destination image")]
    MissingPriorDestination,
    #[error(transparent)]
    C220MaskedAdd(#[from] C220CapturedMaskedAddError),
    #[error(transparent)]
    C310MaskedAdd(#[from] C310CapturedMaskedAddError),
    #[error(transparent)]
    C220Subtract(#[from] C220CapturedSubError),
    #[error(transparent)]
    C310Subtract(#[from] C310CapturedSubError),
    #[error(transparent)]
    C220Multiply(#[from] C220CapturedMulError),
    #[error(transparent)]
    C310Multiply(#[from] C310CapturedMulError),
}

pub fn execute_captured_replay(
    architecture: Architecture,
    operation: CapturedReplayOperation,
    x: &[u8],
    y: &[u8],
    prior_destination: Option<&[u8]>,
) -> Result<CapturedReplay, CapturedReplayError> {
    match (architecture, operation, prior_destination) {
        (Architecture::Dav2201, CapturedReplayOperation::MaskedAdd, Some(prior)) => Ok(
            CapturedReplay::C220MaskedAdd(execute_captured_c220_masked_add(x, y, prior)?),
        ),
        (Architecture::Dav3510, CapturedReplayOperation::MaskedAdd, Some(prior)) => Ok(
            CapturedReplay::C310MaskedAdd(execute_captured_c310_masked_add(x, y, prior)?),
        ),
        (_, CapturedReplayOperation::MaskedAdd, None) => {
            Err(CapturedReplayError::MissingPriorDestination)
        }
        (Architecture::Dav2201, CapturedReplayOperation::Subtract, Some(prior)) => Ok(
            CapturedReplay::C220Subtract(execute_captured_c220_sub(x, y, prior)?),
        ),
        (Architecture::Dav3510, CapturedReplayOperation::Subtract, Some(prior)) => Ok(
            CapturedReplay::C310Subtract(execute_captured_c310_sub(x, y, prior)?),
        ),
        (Architecture::Dav2201, CapturedReplayOperation::Subtract, None) => Ok(
            CapturedReplay::C220Subtract(execute_captured_c220_sub_predecessor_chains(x, y)?),
        ),
        (Architecture::Dav3510, CapturedReplayOperation::Subtract, None) => Ok(
            CapturedReplay::C310Subtract(execute_captured_c310_sub_predecessor_chains(x, y)?),
        ),
        (Architecture::Dav2201, CapturedReplayOperation::Multiply, Some(prior)) => Ok(
            CapturedReplay::C220Multiply(execute_captured_c220_mul(x, y, prior)?),
        ),
        (Architecture::Dav3510, CapturedReplayOperation::Multiply, Some(prior)) => Ok(
            CapturedReplay::C310Multiply(execute_captured_c310_mul(x, y, prior)?),
        ),
        (Architecture::Dav2201, CapturedReplayOperation::Multiply, None) => Ok(
            CapturedReplay::C220Multiply(execute_captured_c220_mul_predecessor_chains(x, y)?),
        ),
        (Architecture::Dav3510, CapturedReplayOperation::Multiply, None) => Ok(
            CapturedReplay::C310Multiply(execute_captured_c310_mul_predecessor_chains(x, y)?),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatches_all_three_operations_on_both_architectures() {
        let x = vec![0; 4096];
        let y = vec![0; 4096];
        let prior = vec![0; 4096];
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            for operation in [
                CapturedReplayOperation::MaskedAdd,
                CapturedReplayOperation::Subtract,
                CapturedReplayOperation::Multiply,
            ] {
                let run =
                    execute_captured_replay(architecture, operation, &x, &y, Some(&prior)).unwrap();
                assert_eq!(run.output().len(), 4096);
                assert_eq!(run.tile_count(), 32);
            }
            for operation in [
                CapturedReplayOperation::Subtract,
                CapturedReplayOperation::Multiply,
            ] {
                let run = execute_captured_replay(architecture, operation, &x, &y, None).unwrap();
                assert_eq!(run.output().len(), 4096);
            }
        }
    }

    #[test]
    fn missing_add_state_and_wrong_length_fail_closed() {
        let bytes = vec![0; 4096];
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            assert!(matches!(
                execute_captured_replay(
                    architecture,
                    CapturedReplayOperation::MaskedAdd,
                    &bytes,
                    &bytes,
                    None,
                ),
                Err(CapturedReplayError::MissingPriorDestination)
            ));
            assert!(
                execute_captured_replay(
                    architecture,
                    CapturedReplayOperation::Subtract,
                    &bytes[..4095],
                    &bytes,
                    None,
                )
                .is_err()
            );
            assert!(
                execute_captured_replay(
                    architecture,
                    CapturedReplayOperation::Multiply,
                    &bytes[..4095],
                    &bytes,
                    None,
                )
                .is_err()
            );
        }
    }
}
