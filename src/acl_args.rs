
use crate::kernel_config::{DecodedKernelConfig, ReplayRunner};
use serde::Serialize;
use thiserror::Error;

const POINTER_BYTES: usize = 8;
const WORKSPACE_EXTRA_BYTES: u64 = 0x100_0000;
const MAX_EXACT_FLOAT_TILING_BYTES: u64 = (1_u64 << 53) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AclArgKind {
    Input,
    Output,
    Tiling,
    Workspace,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AclArgSlot {
    pub index: usize,
    pub kind: AclArgKind,
    pub path: Option<String>,
    pub logical_bytes: Option<u64>,
    pub host_allocation_bytes: Option<u64>,
    pub device_allocation_bytes: Option<u64>,
    pub host_to_device_copy_bytes: Option<u64>,
    pub device_to_host_copy_bytes: Option<u64>,
    pub passes_null_pointer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AclArgumentPlan {
    pub slots: Vec<AclArgSlot>,
    pub append_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AclArgumentPlanError {
    #[error("ACL pointer argument planning requires old_mode other than 1")]
    LegacyRunner,
    #[error("{kind:?} argument {index} has no byte size")]
    MissingSize { kind: AclArgKind, index: usize },
    #[error("{kind:?} argument {index} allocation size overflows u64")]
    AllocationOverflow { kind: AclArgKind, index: usize },
    #[error("tiling size {size} exceeds the exactly modeled floating-point conversion range")]
    TilingRoundingUnproven { size: u64 },
    #[error("argument pointer array byte length overflows usize")]
    PointerArrayOverflow,
}

impl AclArgumentPlan {
    pub fn from_config(config: &DecodedKernelConfig) -> Result<Self, AclArgumentPlanError> {
        if config.runner != ReplayRunner::Acl {
            return Err(AclArgumentPlanError::LegacyRunner);
        }
        let mut slots = Vec::new();
        for input in &config.inputs {
            let index = slots.len();
            let size = if input.is_null_input {
                input.size
            } else {
                Some(input.size.ok_or(AclArgumentPlanError::MissingSize {
                    kind: AclArgKind::Input,
                    index,
                })?)
            };
            slots.push(AclArgSlot {
                index,
                kind: AclArgKind::Input,
                path: Some(input.path.clone()),
                logical_bytes: size,
                host_allocation_bytes: (!input.is_null_input).then_some(size.unwrap_or(0)),
                device_allocation_bytes: (!input.is_null_input).then_some(size.unwrap_or(0)),
                host_to_device_copy_bytes: (!input.is_null_input).then_some(size.unwrap_or(0)),
                device_to_host_copy_bytes: None,
                passes_null_pointer: input.is_null_input,
            });
        }
        for output in &config.outputs {
            let index = slots.len();
            let size = output.size.ok_or(AclArgumentPlanError::MissingSize {
                kind: AclArgKind::Output,
                index,
            })?;
            slots.push(AclArgSlot {
                index,
                kind: AclArgKind::Output,
                path: Some(output.path.clone()),
                logical_bytes: Some(size),
                host_allocation_bytes: Some(size),
                device_allocation_bytes: Some(size),
                host_to_device_copy_bytes: None,
                device_to_host_copy_bytes: Some(size),
                passes_null_pointer: false,
            });
        }
        if let Some(tiling) = &config.tiling_data {
            let index = slots.len();
            let size = tiling.size;
            if size > MAX_EXACT_FLOAT_TILING_BYTES {
                return Err(AclArgumentPlanError::TilingRoundingUnproven { size });
            }
            let rounded = size
                .checked_add(31)
                .map(|value| value & !31)
                .and_then(|value| value.checked_add(32))
                .ok_or(AclArgumentPlanError::AllocationOverflow {
                    kind: AclArgKind::Tiling,
                    index,
                })?;
            slots.push(AclArgSlot {
                index,
                kind: AclArgKind::Tiling,
                path: Some(tiling.path.clone()),
                logical_bytes: Some(size),
                host_allocation_bytes: Some(size),
                device_allocation_bytes: Some(rounded),
                host_to_device_copy_bytes: Some(size),
                device_to_host_copy_bytes: None,
                passes_null_pointer: false,
            });
        }
        for &size in &config.workspace_sizes {
            let index = slots.len();
            let allocation = size.checked_add(WORKSPACE_EXTRA_BYTES).ok_or(
                AclArgumentPlanError::AllocationOverflow {
                    kind: AclArgKind::Workspace,
                    index,
                },
            )?;
            slots.push(AclArgSlot {
                index,
                kind: AclArgKind::Workspace,
                path: None,
                logical_bytes: Some(size),
                host_allocation_bytes: None,
                device_allocation_bytes: Some(allocation),
                host_to_device_copy_bytes: None,
                device_to_host_copy_bytes: None,
                passes_null_pointer: false,
            });
        }
        let append_bytes = slots
            .len()
            .checked_mul(POINTER_BYTES)
            .ok_or(AclArgumentPlanError::PointerArrayOverflow)?;
        Ok(Self {
            slots,
            append_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_config::KernelConfigDocument;

    fn config(json: &[u8]) -> DecodedKernelConfig {
        KernelConfigDocument::from_slice(json)
            .unwrap()
            .decode()
            .unwrap()
    }

    #[test]
    fn preserves_pointer_slots_and_vendor_allocation_requests() {
        let config = config(
            br#"{"old_mode":"0","input_path":"n;input.bin","input_size":"999;9","output_name":"out.bin","output_size":"64","tiling_data_path":"tile.bin;96","workspace_size":"128"}"#,
        );
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        assert_eq!(plan.slots.len(), 5);
        assert_eq!(plan.append_bytes, 40);
        assert_eq!(plan.slots[0].kind, AclArgKind::Input);
        assert!(plan.slots[0].passes_null_pointer);
        assert_eq!(plan.slots[0].device_allocation_bytes, None);
        assert_eq!(plan.slots[1].device_allocation_bytes, Some(9));
        assert_eq!(plan.slots[1].host_to_device_copy_bytes, Some(9));
        assert_eq!(plan.slots[2].kind, AclArgKind::Output);
        assert_eq!(plan.slots[2].device_to_host_copy_bytes, Some(64));
        assert_eq!(plan.slots[3].kind, AclArgKind::Tiling);
        assert_eq!(plan.slots[3].device_allocation_bytes, Some(128));
        assert_eq!(plan.slots[3].host_to_device_copy_bytes, Some(96));
        assert_eq!(plan.slots[4].kind, AclArgKind::Workspace);
        assert_eq!(plan.slots[4].device_allocation_bytes, Some(0x100_0080));
    }

    #[test]
    fn tiling_reserves_one_extra_32_byte_block_after_round_up() {
        for (size, allocated) in [(0, 32), (1, 64), (32, 64), (33, 96)] {
            let config =
                config(format!(r#"{{"old_mode":"0","tiling_data_path":"t;{size}"}}"#).as_bytes());
            let plan = AclArgumentPlan::from_config(&config).unwrap();
            assert_eq!(plan.slots[0].device_allocation_bytes, Some(allocated));
        }
    }

    #[test]
    fn null_input_without_size_still_occupies_one_pointer_slot() {
        let config = config(br#"{"old_mode":"0","input_path":"n"}"#);
        let plan = AclArgumentPlan::from_config(&config).unwrap();
        assert_eq!(plan.append_bytes, 8);
        assert_eq!(plan.slots[0].logical_bytes, None);
        assert!(plan.slots[0].passes_null_pointer);
        assert_eq!(plan.slots[0].device_allocation_bytes, None);
    }

    #[test]
    fn fails_closed_for_legacy_missing_sizes_and_unproven_tiling_precision() {
        assert_eq!(
            AclArgumentPlan::from_config(&config(br#"{}"#)),
            Err(AclArgumentPlanError::LegacyRunner)
        );
        assert_eq!(
            AclArgumentPlan::from_config(&config(br#"{"old_mode":"0","input_path":"a"}"#)),
            Err(AclArgumentPlanError::MissingSize {
                kind: AclArgKind::Input,
                index: 0
            })
        );
        assert!(matches!(
            AclArgumentPlan::from_config(&config(
                br#"{"old_mode":"0","tiling_data_path":"t;9007199254740992"}"#
            )),
            Err(AclArgumentPlanError::TilingRoundingUnproven { .. })
        ));
        assert!(matches!(
            AclArgumentPlan::from_config(&config(
                br#"{"old_mode":"0","workspace_size":"18446744073709551615"}"#
            )),
            Err(AclArgumentPlanError::AllocationOverflow {
                kind: AclArgKind::Workspace,
                ..
            })
        ));
    }
}
