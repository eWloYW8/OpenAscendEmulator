
use crate::acl_args::{AclArgKind, AclArgumentPlan, AclArgumentPlanError};
use crate::kernel_config::DecodedKernelConfig;
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SeedArgument {
    Null,
    Region(usize),
}

pub struct SeedRegion {
    kind: AclArgKind,
    allocation_bytes: u64,
    known_prefix: Vec<u8>,
}

impl SeedRegion {
    pub const fn kind(&self) -> AclArgKind {
        self.kind
    }

    pub const fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    pub fn known_prefix(&self) -> &[u8] {
        &self.known_prefix
    }

    pub fn unknown_tail_bytes(&self) -> u64 {
        self.allocation_bytes - self.known_prefix.len() as u64
    }
}

pub struct ReplaySeed {
    pub arguments: Vec<SeedArgument>,
    pub regions: Vec<SeedRegion>,
    pub append_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SeedRegionSummary {
    pub index: usize,
    pub kind: AclArgKind,
    pub allocation_bytes: u64,
    pub known_prefix_bytes: usize,
    pub unknown_tail_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplaySeedSummary {
    pub arguments: Vec<SeedArgument>,
    pub regions: Vec<SeedRegionSummary>,
    pub append_bytes: usize,
}

#[derive(Debug, Error)]
pub enum ReplaySeedError {
    #[error(transparent)]
    Arguments(#[from] AclArgumentPlanError),
    #[error("argument {index} has no source path")]
    MissingPath { index: usize },
    #[error("argument {index} has no known input byte count")]
    MissingSize { index: usize },
    #[error("loaded bytes exceed the configured limit of {limit} bytes")]
    LoadLimitExceeded { limit: u64 },
    #[error("argument {index} byte count cannot fit in host memory")]
    HostSizeOverflow { index: usize },
    #[error("cannot reserve {requested} host bytes for argument {index}")]
    HostAllocationFailed { index: usize, requested: u64 },
    #[error("failed to read argument {index} from {path}: {source}")]
    FileIo {
        index: usize,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("argument {index} at {path} declares {expected} bytes but file has {actual}")]
    FileSizeMismatch {
        index: usize,
        path: PathBuf,
        expected: u64,
        actual: u64,
    },
    #[error("argument {index} at {path} grew while being read")]
    FileChangedDuringRead { index: usize, path: PathBuf },
    #[error("argument {index} has more initialized bytes than its device allocation")]
    InitializedBytesExceedAllocation { index: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SeedReadError {
    #[error("region {0} does not exist")]
    InvalidRegion(usize),
    #[error("region {region} read range overflows")]
    RangeOverflow { region: usize },
    #[error(
        "region {region} read [{offset}, {end}) exceeds allocation of {allocation_bytes} bytes"
    )]
    OutOfBounds {
        region: usize,
        offset: usize,
        end: usize,
        allocation_bytes: u64,
    },
    #[error("region {region} includes bytes not known from the replay files")]
    UnknownBytes { region: usize },
}

impl ReplaySeed {
    pub fn summary(&self) -> ReplaySeedSummary {
        ReplaySeedSummary {
            arguments: self.arguments.clone(),
            regions: self
                .regions
                .iter()
                .enumerate()
                .map(|(index, region)| SeedRegionSummary {
                    index,
                    kind: region.kind(),
                    allocation_bytes: region.allocation_bytes(),
                    known_prefix_bytes: region.known_prefix().len(),
                    unknown_tail_bytes: region.unknown_tail_bytes(),
                })
                .collect(),
            append_bytes: self.append_bytes,
        }
    }

    pub fn load(
        config: &DecodedKernelConfig,
        max_loaded_bytes: u64,
    ) -> Result<Self, ReplaySeedError> {
        let plan = AclArgumentPlan::from_config(config)?;
        let mut arguments = Vec::with_capacity(plan.slots.len());
        let mut regions = Vec::with_capacity(plan.slots.len());
        let mut total_loaded_bytes = 0_u64;
        for slot in plan.slots {
            if slot.passes_null_pointer {
                arguments.push(SeedArgument::Null);
                continue;
            }
            let allocation_bytes = slot
                .device_allocation_bytes
                .expect("every non-null planned slot has a device allocation");
            let known_prefix = if matches!(slot.kind, AclArgKind::Input | AclArgKind::Tiling) {
                let size = slot
                    .logical_bytes
                    .ok_or(ReplaySeedError::MissingSize { index: slot.index })?;
                total_loaded_bytes = total_loaded_bytes
                    .checked_add(size)
                    .filter(|total| *total <= max_loaded_bytes)
                    .ok_or(ReplaySeedError::LoadLimitExceeded {
                        limit: max_loaded_bytes,
                    })?;
                let path = slot
                    .path
                    .as_deref()
                    .ok_or(ReplaySeedError::MissingPath { index: slot.index })?;
                read_exact_file(Path::new(path), size, slot.index)?
            } else {
                Vec::new()
            };
            if known_prefix.len() as u64 > allocation_bytes {
                return Err(ReplaySeedError::InitializedBytesExceedAllocation {
                    index: slot.index,
                });
            }
            arguments.push(SeedArgument::Region(regions.len()));
            regions.push(SeedRegion {
                kind: slot.kind,
                allocation_bytes,
                known_prefix,
            });
        }
        Ok(Self {
            arguments,
            regions,
            append_bytes: plan.append_bytes,
        })
    }

    pub fn read_known(
        &self,
        region: usize,
        offset: usize,
        len: usize,
    ) -> Result<&[u8], SeedReadError> {
        let memory = self
            .regions
            .get(region)
            .ok_or(SeedReadError::InvalidRegion(region))?;
        let end = offset
            .checked_add(len)
            .ok_or(SeedReadError::RangeOverflow { region })?;
        if !u64::try_from(end).is_ok_and(|end| end <= memory.allocation_bytes()) {
            return Err(SeedReadError::OutOfBounds {
                region,
                offset,
                end,
                allocation_bytes: memory.allocation_bytes(),
            });
        }
        memory
            .known_prefix()
            .get(offset..end)
            .ok_or(SeedReadError::UnknownBytes { region })
    }
}

fn read_exact_file(path: &Path, expected: u64, index: usize) -> Result<Vec<u8>, ReplaySeedError> {
    let mut file = File::open(path).map_err(|source| ReplaySeedError::FileIo {
        index,
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = file.metadata().map_err(|source| ReplaySeedError::FileIo {
        index,
        path: path.to_path_buf(),
        source,
    })?;
    if metadata.len() != expected {
        return Err(ReplaySeedError::FileSizeMismatch {
            index,
            path: path.to_path_buf(),
            expected,
            actual: metadata.len(),
        });
    }
    let expected_usize =
        usize::try_from(expected).map_err(|_| ReplaySeedError::HostSizeOverflow { index })?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_usize)
        .map_err(|_| ReplaySeedError::HostAllocationFailed {
            index,
            requested: expected,
        })?;
    bytes.resize(expected_usize, 0);
    file.read_exact(&mut bytes)
        .map_err(|source| ReplaySeedError::FileIo {
            index,
            path: path.to_path_buf(),
            source,
        })?;
    let mut extra = [0_u8; 1];
    let extra_count = file
        .read(&mut extra)
        .map_err(|source| ReplaySeedError::FileIo {
            index,
            path: path.to_path_buf(),
            source,
        })?;
    if extra_count != 0 {
        return Err(ReplaySeedError::FileChangedDuringRead {
            index,
            path: path.to_path_buf(),
        });
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_config::KernelConfigDocument;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "open-ascend-emulator-seed-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn config(json: &str) -> DecodedKernelConfig {
        KernelConfigDocument::from_slice(json.as_bytes())
            .unwrap()
            .decode()
            .unwrap()
    }

    #[test]
    fn stages_known_input_and_tiling_without_inventing_output_or_padding() {
        let tree = TempTree::new();
        let input = tree.0.join("input.bin");
        let tiling = tree.0.join("tile.bin");
        fs::write(&input, [1_u8, 2, 3]).unwrap();
        fs::write(&tiling, [4_u8, 5]).unwrap();
        let config = config(&format!(
            "{{\"old_mode\":\"0\",\"input_path\":\"n;{}\",\"input_size\":\"999;3\",\"output_name\":\"out.bin\",\"output_size\":\"4\",\"tiling_data_path\":\"{};2\",\"workspace_size\":\"16\"}}",
            input.display(),
            tiling.display()
        ));
        let seed = ReplaySeed::load(&config, 5).unwrap();
        assert_eq!(seed.append_bytes, 40);
        assert_eq!(
            seed.arguments,
            [
                SeedArgument::Null,
                SeedArgument::Region(0),
                SeedArgument::Region(1),
                SeedArgument::Region(2),
                SeedArgument::Region(3)
            ]
        );
        assert_eq!(seed.regions[0].known_prefix(), [1, 2, 3]);
        assert_eq!(seed.regions[0].unknown_tail_bytes(), 0);
        assert_eq!(seed.regions[1].kind(), AclArgKind::Output);
        assert!(seed.regions[1].known_prefix().is_empty());
        assert_eq!(seed.regions[1].unknown_tail_bytes(), 4);
        assert_eq!(seed.regions[2].kind(), AclArgKind::Workspace);
        assert_eq!(seed.regions[2].unknown_tail_bytes(), 0x100_0010);
        assert_eq!(seed.regions[3].kind(), AclArgKind::Tiling);
        assert_eq!(seed.regions[3].known_prefix(), [4, 5]);
        assert_eq!(seed.regions[3].unknown_tail_bytes(), 62);
        let summary = seed.summary();
        assert_eq!(summary.regions[3].known_prefix_bytes, 2);
        assert_eq!(summary.regions[3].unknown_tail_bytes, 62);
        assert_eq!(seed.read_known(0, 1, 2).unwrap(), [2, 3]);
        assert_eq!(seed.read_known(3, 0, 2).unwrap(), [4, 5]);
        assert_eq!(
            seed.read_known(3, 2, 1),
            Err(SeedReadError::UnknownBytes { region: 3 })
        );
        assert_eq!(
            seed.read_known(1, 0, 1),
            Err(SeedReadError::UnknownBytes { region: 1 })
        );
        assert_eq!(
            seed.read_known(0, 3, 1),
            Err(SeedReadError::OutOfBounds {
                region: 0,
                offset: 3,
                end: 4,
                allocation_bytes: 3,
            })
        );
    }

    #[test]
    fn rejects_size_mismatch_and_loads_no_unbounded_bytes() {
        let tree = TempTree::new();
        let input = tree.0.join("input.bin");
        fs::write(&input, [1_u8, 2, 3]).unwrap();
        let config = config(&format!(
            "{{\"old_mode\":\"0\",\"input_path\":\"{}\",\"input_size\":\"4\"}}",
            input.display()
        ));
        assert!(matches!(
            ReplaySeed::load(&config, 4),
            Err(ReplaySeedError::FileSizeMismatch {
                expected: 4,
                actual: 3,
                ..
            })
        ));
        assert!(matches!(
            ReplaySeed::load(&config, 2),
            Err(ReplaySeedError::LoadLimitExceeded { limit: 2 })
        ));
    }
}
