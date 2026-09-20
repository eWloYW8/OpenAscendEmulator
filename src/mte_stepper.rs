use serde::Serialize;
use thiserror::Error;

use crate::acl_address_space::AclReplayAddressSpace;
use crate::architecture::Architecture;
use crate::machine::{ScalarInstructionError, ScalarMemoryBus};
use crate::mte_c220::{
    C220MovOutToUbDescriptor, C220MovOutToUbError, CAPTURED_C220_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD,
    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD, CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
};
use crate::mte_c310::{
    C310_ADD_MOV_ALIGN_X_WORD, C310_ADD_MOV_ALIGN_Y_WORD, C310_TILING_MOV_ALIGN_WORD,
    C310CapturedMovAlignDecode, C310CapturedMovAlignError, C310MovAlignRegisterSelectors,
    C310TilingMovAlignRegisters,
};
use crate::stepper::{ScalarProgramStep, ScalarStepper};
use crate::ub_replay::{UbReplayError, UbReplayMemory, UbTransferResult};

pub const MTE2_TO_SCALAR_SET_FLAG0_WORD: u32 = 0x40a0_1000;
pub const MTE2_TO_SCALAR_WAIT_FLAG0_WORD: u32 = 0x40c0_1000;
pub const MAX_PENDING_MTE2_TRANSFERS: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq)]
enum PendingMte2 {
    C220 {
        descriptor: C220MovOutToUbDescriptor,
        source_address: u64,
        destination_address: u64,
    },
    C310(C310CapturedMovAlignDecode),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MteCoreStepper {
    scalar: ScalarStepper,
    ub: UbReplayMemory,
    pending_mte2: Vec<PendingMte2>,
    flag0_set: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum MteAction {
    Issue {
        source_address: u64,
        destination_address: u64,
        planned_bytes: usize,
        pending_count: usize,
    },
    SetFlag {
        pending_count: usize,
    },
    WaitFlag {
        transfers: Vec<UbTransferResult>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MteProgramStep {
    pub pc: u64,
    pub word: u32,
    pub next_pc: u64,
    pub action: MteAction,
}

#[derive(Debug, Error)]
pub enum MteStepperError {
    #[error("program ended before MTE word at PC {pc:#x}")]
    ProgramEnded { pc: u64 },
    #[error("word {word:#010x} at PC {pc:#x} is not an implemented MTE2 or flag operation")]
    UnsupportedWord { pc: u64, word: u32 },
    #[error("SPR {index} is unavailable for MTE word at PC {pc:#x}")]
    MissingSpr { pc: u64, index: u16 },
    #[error("MTE2 pending count exceeds {MAX_PENDING_MTE2_TRANSFERS}")]
    PendingLimit,
    #[error("MTE2-to-Scalar flag 0 has no pending transfer")]
    SetWithoutTransfer,
    #[error("MTE2-to-Scalar flag 0 is already set")]
    FlagAlreadySet,
    #[error("MTE2-to-Scalar flag 0 was not set before wait")]
    WaitWithoutFlag,
    #[error("MTE2 transfer byte count overflows usize")]
    TransferSizeOverflow,
    #[error(transparent)]
    C220(#[from] C220MovOutToUbError),
    #[error(transparent)]
    C310(#[from] C310CapturedMovAlignError),
    #[error(transparent)]
    Ub(#[from] UbReplayError),
}

impl MteCoreStepper {
    pub fn new(scalar: ScalarStepper, ub: UbReplayMemory) -> Self {
        Self {
            scalar,
            ub,
            pending_mte2: Vec::new(),
            flag0_set: false,
        }
    }

    pub const fn scalar(&self) -> &ScalarStepper {
        &self.scalar
    }

    pub fn scalar_mut(&mut self) -> &mut ScalarStepper {
        &mut self.scalar
    }

    pub const fn ub(&self) -> &UbReplayMemory {
        &self.ub
    }

    pub const fn pending_mte2_count(&self) -> usize {
        self.pending_mte2.len()
    }

    pub const fn flag0_set(&self) -> bool {
        self.flag0_set
    }

    pub fn step_scalar_word<B: ScalarMemoryBus>(
        &mut self,
        word: u32,
        bus: &mut B,
    ) -> Result<ScalarProgramStep, ScalarInstructionError<B::Error>> {
        self.scalar.step_word(word, bus)
    }

    pub fn step_mte_word(
        &mut self,
        word: u32,
        source: &AclReplayAddressSpace,
    ) -> Result<MteProgramStep, MteStepperError> {
        let pc = self.scalar.pc();
        if self.scalar.is_halted() {
            return Err(MteStepperError::ProgramEnded { pc });
        }
        let action = match word {
            MTE2_TO_SCALAR_SET_FLAG0_WORD => {
                if self.flag0_set {
                    return Err(MteStepperError::FlagAlreadySet);
                }
                if self.pending_mte2.is_empty() {
                    return Err(MteStepperError::SetWithoutTransfer);
                }
                self.flag0_set = true;
                MteAction::SetFlag {
                    pending_count: self.pending_mte2.len(),
                }
            }
            MTE2_TO_SCALAR_WAIT_FLAG0_WORD => {
                if !self.flag0_set {
                    return Err(MteStepperError::WaitWithoutFlag);
                }
                let mut staged = self.ub.clone();
                let mut transfers = Vec::with_capacity(self.pending_mte2.len());
                for transfer in &self.pending_mte2 {
                    let result = match *transfer {
                        PendingMte2::C220 {
                            descriptor,
                            source_address,
                            destination_address,
                        } => staged.copy_c220_mov_out_to_ub(
                            source,
                            descriptor,
                            source_address,
                            destination_address,
                        )?,
                        PendingMte2::C310(decoded) => {
                            staged.copy_c310_mov_align_hbm_to_ub(source, decoded)?
                        }
                    };
                    transfers.push(result);
                }
                self.ub = staged;
                self.pending_mte2.clear();
                self.flag0_set = false;
                MteAction::WaitFlag { transfers }
            }
            _ => {
                if self.pending_mte2.len() >= MAX_PENDING_MTE2_TRANSFERS {
                    return Err(MteStepperError::PendingLimit);
                }
                if self.flag0_set {
                    return Err(MteStepperError::FlagAlreadySet);
                }
                let (transfer, source_address, destination_address, planned_bytes) =
                    self.decode_transfer(pc, word)?;
                self.pending_mte2.push(transfer);
                MteAction::Issue {
                    source_address,
                    destination_address,
                    planned_bytes,
                    pending_count: self.pending_mte2.len(),
                }
            }
        };
        self.scalar.advance_sequential();
        Ok(MteProgramStep {
            pc,
            word,
            next_pc: self.scalar.pc(),
            action,
        })
    }

    fn decode_transfer(
        &self,
        pc: u64,
        word: u32,
    ) -> Result<(PendingMte2, u64, u64, usize), MteStepperError> {
        let machine = self.scalar.machine();
        let x = machine.xregs();
        match machine.architecture() {
            Architecture::Dav2201 => {
                let (destination, source, descriptor) = match word {
                    CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD => (0, 1, 2),
                    CAPTURED_C220_MOV_OUT_TO_UB_X_WORD => (15, 19, 3),
                    CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD => (18, 15, 3),
                    CAPTURED_C220_SUB_MOV_OUT_TO_UB_X_WORD => (13, 17, 4),
                    CAPTURED_C220_SUB_MOV_OUT_TO_UB_Y_WORD => (16, 13, 4),
                    _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
                };
                let descriptor = C220MovOutToUbDescriptor::decode(word, x[descriptor])?;
                let source_address = x[source];
                let destination_address = x[destination];
                let planned_bytes = descriptor
                    .segments(source_address, destination_address)?
                    .len()
                    .checked_mul(32)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                Ok((
                    PendingMte2::C220 {
                        descriptor,
                        source_address,
                        destination_address,
                    },
                    source_address,
                    destination_address,
                    planned_bytes,
                ))
            }
            Architecture::Dav3510 => {
                let spr = |index| {
                    machine
                        .spr_value(index)
                        .ok_or(MteStepperError::MissingSpr { pc, index })
                };
                let decoded = match word {
                    C310_TILING_MOV_ALIGN_WORD => C310TilingMovAlignRegisters {
                        source_xreg1: x[1],
                        shape_xreg4: x[4],
                        destination_and_stride_xreg7: x[7],
                        loop_spr105: spr(105)?,
                        inner_stride_spr106: spr(106)?,
                        outer_stride_spr107: spr(107)?,
                    }
                    .decode(word)?,
                    C310_ADD_MOV_ALIGN_X_WORD
                    | C310_ADD_MOV_ALIGN_Y_WORD
                    | 0x74ad_8bae
                    | 0x74b3_6bae => C310MovAlignRegisterSelectors::from_captured_word(word)?
                        .capture(x, spr(105)?, spr(106)?, spr(107)?)
                        .decode_hbm_to_ub_word(word)?,
                    _ => return Err(MteStepperError::UnsupportedWord { pc, word }),
                };
                let coordinates = decoded
                    .parameters
                    .coordinates()
                    .map_err(UbReplayError::from)?;
                let planned_bytes = coordinates
                    .len()
                    .checked_mul(decoded.burst_bytes as usize)
                    .ok_or(MteStepperError::TransferSizeOverflow)?;
                Ok((
                    PendingMte2::C310(decoded),
                    decoded.parameters.source_base,
                    decoded.parameters.destination_base,
                    planned_bytes,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acl_address_space::AclArgumentImage;
    use crate::acl_args::AclArgumentPlan;
    use crate::addressed_replay::AddressedReplayMemory;
    use crate::kernel_config::KernelConfigDocument;
    use crate::machine::ScalarMachine;
    use crate::replay_memory::{MemoryByteState, ReplayMemory};
    use crate::replay_seed::ReplaySeed;
    use crate::rvec::{C310_CAPTURED_VLDI_V0_WORD, C310RvecValueMachine};
    use std::fs;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        directory: PathBuf,
        source: AclReplayAddressSpace,
    }

    impl Fixture {
        fn new(bytes: &[u8], tiling: bool) -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "open-ascend-mte-stepper-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&directory).unwrap();
            let path = directory.join("input.bin");
            fs::write(&path, bytes).unwrap();
            let json = if tiling {
                format!(
                    "{{\"old_mode\":\"0\",\"output_name\":\"z.bin\",\"output_size\":\"32\",\"tiling_data_path\":\"{};{}\"}}",
                    path.display(),
                    bytes.len(),
                )
            } else {
                format!(
                    "{{\"old_mode\":\"0\",\"input_path\":\"{}\",\"input_size\":\"{}\",\"output_name\":\"z.bin\",\"output_size\":\"32\"}}",
                    path.display(),
                    bytes.len(),
                )
            };
            let config = KernelConfigDocument::from_slice(json.as_bytes())
                .unwrap()
                .decode()
                .unwrap();
            let plan = AclArgumentPlan::from_config(&config).unwrap();
            let seed = ReplaySeed::load(&config, bytes.len() as u64).unwrap();
            let memory = ReplayMemory::new(seed, 64, 128);
            let pointers: &[u64] = if tiling {
                &[0x2000, 0x3000]
            } else {
                &[0x3000, 0x2000]
            };
            let regions = AddressedReplayMemory::bind(memory, pointers).unwrap();
            let image = AclArgumentImage::new(&plan, 0x1000, pointers, &[]).unwrap();
            let source = AclReplayAddressSpace::new(image, regions).unwrap();
            Self { directory, source }
        }

        fn two_inputs(x: &[u8], y: &[u8]) -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let directory = std::env::temp_dir().join(format!(
                "open-ascend-mte-stepper-{}-{id}",
                std::process::id()
            ));
            fs::create_dir(&directory).unwrap();
            let x_path = directory.join("x.bin");
            let y_path = directory.join("y.bin");
            fs::write(&x_path, x).unwrap();
            fs::write(&y_path, y).unwrap();
            let json = format!(
                "{{\"old_mode\":\"0\",\"input_path\":\"{};{}\",\"input_size\":\"{};{}\",\"output_name\":\"z.bin\",\"output_size\":\"32\"}}",
                x_path.display(),
                y_path.display(),
                x.len(),
                y.len(),
            );
            let config = KernelConfigDocument::from_slice(json.as_bytes())
                .unwrap()
                .decode()
                .unwrap();
            let plan = AclArgumentPlan::from_config(&config).unwrap();
            let seed = ReplaySeed::load(&config, 4096).unwrap();
            let memory = ReplayMemory::new(seed, 512, 512);
            let pointers = &[0x3000, 0x4000, 0x2000];
            let regions = AddressedReplayMemory::bind(memory, pointers).unwrap();
            let image = AclArgumentImage::new(&plan, 0x1000, pointers, &[]).unwrap();
            let source = AclReplayAddressSpace::new(image, regions).unwrap();
            Self { directory, source }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.directory);
        }
    }

    fn stepper(architecture: Architecture, source_address: u64) -> MteCoreStepper {
        let mut machine = ScalarMachine::from_pem_initial_state(architecture);
        machine.set_xreg(1, source_address).unwrap();
        match architecture {
            Architecture::Dav2201 => machine.set_xreg(2, 0x10010).unwrap(),
            Architecture::Dav3510 => machine.set_xreg(4, 0x4000_0010).unwrap(),
        }
        MteCoreStepper::new(
            ScalarStepper::new(machine, 0x1000),
            UbReplayMemory::new(512, 256),
        )
    }

    struct NoMemoryBus;

    impl ScalarMemoryBus for NoMemoryBus {
        type Error = io::Error;

        fn read(&mut self, _address: u64, _destination: &mut [u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory read was not expected"))
        }

        fn write(&mut self, _address: u64, _source: &[u8]) -> Result<(), Self::Error> {
            Err(io::Error::other("memory write was not expected"))
        }
    }

    #[test]
    fn first_mte_transfer_becomes_visible_only_after_matching_flag_wait() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, word) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            ),
            (Architecture::Dav3510, C310_TILING_MOV_ALIGN_WORD),
        ] {
            let mut core = stepper(architecture, 0x3000);
            let issue = core.step_mte_word(word, &fixture.source).unwrap();
            assert_eq!(issue.pc, 0x1000);
            assert_eq!(issue.next_pc, 0x1004);
            assert!(matches!(
                issue.action,
                MteAction::Issue {
                    source_address: 0x3000,
                    destination_address: 0,
                    planned_bytes: 32,
                    pending_count: 1,
                }
            ));
            assert_eq!(core.ub().tracked_bytes(), 0);
            assert_eq!(core.pending_mte2_count(), 1);

            let signal = core
                .step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(signal.pc, 0x1004);
            assert!(matches!(
                signal.action,
                MteAction::SetFlag { pending_count: 1 }
            ));
            assert!(core.flag0_set());
            assert_eq!(core.ub().tracked_bytes(), 0);

            let wait = core
                .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(wait.pc, 0x1008);
            assert_eq!(wait.next_pc, 0x100c);
            assert_eq!(
                wait.action,
                MteAction::WaitFlag {
                    transfers: vec![UbTransferResult {
                        segment_count: 1,
                        bytes: 32,
                        known_bytes: 4,
                        unknown_bytes: 28,
                    }]
                }
            );
            assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
            assert_eq!(
                core.ub().read_states(4, 28).unwrap(),
                [MemoryByteState::Unknown; 28]
            );
            assert_eq!(core.pending_mte2_count(), 0);
            assert!(!core.flag0_set());
        }
    }

    #[test]
    fn scalar_words_interleave_with_mte_issue_signal_and_wait() {
        let fixture = Fixture::new(&1024_u32.to_le_bytes(), true);
        for (architecture, word) in [
            (
                Architecture::Dav2201,
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD,
            ),
            (Architecture::Dav3510, C310_TILING_MOV_ALIGN_WORD),
        ] {
            let mut core = stepper(architecture, 0x3000);
            let mut bus = NoMemoryBus;
            core.step_mte_word(word, &fixture.source).unwrap();
            assert_eq!(
                core.step_scalar_word(0x0706_0001, &mut bus).unwrap().pc,
                0x1004
            );
            assert_eq!(core.ub().tracked_bytes(), 0);
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(
                core.step_scalar_word(0x0708_0002, &mut bus).unwrap().pc,
                0x100c
            );
            assert_eq!(core.ub().tracked_bytes(), 0);
            let wait = core
                .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
                .unwrap();
            assert_eq!(wait.pc, 0x1010);
            assert_eq!(core.scalar().pc(), 0x1014);
            assert_eq!(core.ub().read_known(0, 4).unwrap(), 1024_u32.to_le_bytes());
        }
    }

    #[test]
    fn invalid_flag_order_and_failed_source_keep_pc_and_ub_unchanged() {
        let fixture = Fixture::new(&[0x5a; 32], false);
        let mut core = stepper(Architecture::Dav2201, 0x4000);
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::SetWithoutTransfer)
        ));
        assert_eq!(core.scalar().pc(), 0x1000);
        core.step_mte_word(CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD, &fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::WaitWithoutFlag)
        ));
        assert_eq!(core, before);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        let before = core.clone();
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::Ub(UbReplayError::Source(_)))
        ));
        assert_eq!(core, before);
        assert!(matches!(
            core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source),
            Err(MteStepperError::FlagAlreadySet)
        ));
        assert_eq!(core, before);
    }

    #[test]
    fn c220_registers_are_captured_at_issue_for_multiple_transfers() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav2201, 0x3000);
        core.scalar_mut()
            .machine_mut()
            .set_xreg(3, 0x40010)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(15, 0).unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(19, 0x3000)
            .unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_X_WORD, &fixture.source)
            .unwrap();
        core.scalar_mut()
            .machine_mut()
            .set_xreg(15, 0x3000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(18, 0x80).unwrap();
        core.step_mte_word(CAPTURED_C220_MOV_OUT_TO_UB_Y_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.pending_mte2_count(), 2);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        let wait = core
            .step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            wait.action,
            MteAction::WaitFlag { transfers } if transfers.len() == 2
                && transfers.iter().all(|transfer| transfer.bytes == 128)
        ));
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c310_input_transfers_commit_both_windows_on_one_wait() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(22, 0).unwrap();
        machine.set_xreg(24, 0x3000).unwrap();
        machine.set_xreg(23, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(11, 0x0000_8000_0000_0080).unwrap();
        let first = core.step_mte_word(0x74ad_8bae, &fixture.source).unwrap();
        assert!(matches!(
            first.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0,
                planned_bytes: 128,
                pending_count: 1,
            }
        ));
        core.scalar_mut()
            .machine_mut()
            .set_xreg(22, 0x3000)
            .unwrap();
        core.scalar_mut().machine_mut().set_xreg(25, 0x80).unwrap();
        let second = core.step_mte_word(0x74b3_6bae, &fixture.source).unwrap();
        assert!(matches!(
            second.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0x80,
                planned_bytes: 128,
                pending_count: 2,
            }
        ));
        assert_eq!(core.ub().tracked_bytes(), 0);
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c310_add_input_words_use_distinct_register_selectors() {
        let input: Vec<u8> = (0..128).collect();
        let fixture = Fixture::new(&input, false);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(25, 0).unwrap();
        machine.set_xreg(0, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(8, 0x0000_8000_0000_0080).unwrap();
        let x_issue = core
            .step_mte_word(C310_ADD_MOV_ALIGN_X_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            x_issue.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0,
                planned_bytes: 128,
                pending_count: 1,
            }
        ));
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(0, 0x3000).unwrap();
        machine.set_xreg(1, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(2, 0x80).unwrap();
        let y_issue = core
            .step_mte_word(C310_ADD_MOV_ALIGN_Y_WORD, &fixture.source)
            .unwrap();
        assert!(matches!(
            y_issue.action,
            MteAction::Issue {
                source_address: 0x3000,
                destination_address: 0x80,
                planned_bytes: 128,
                pending_count: 2,
            }
        ));
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), input);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), input);
    }

    #[test]
    fn c310_add_mte2_windows_feed_the_captured_vldi_v0_image() {
        let x = (0..128).collect::<Vec<u8>>();
        let y = (0..128).map(|value| 255 - value).collect::<Vec<u8>>();
        let fixture = Fixture::two_inputs(&x, &y);
        let mut core = stepper(Architecture::Dav3510, 0x3000);
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(25, 0).unwrap();
        machine.set_xreg(0, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(8, 0x0000_8000_0000_0080).unwrap();
        core.step_mte_word(C310_ADD_MOV_ALIGN_X_WORD, &fixture.source)
            .unwrap();
        let machine = core.scalar_mut().machine_mut();
        machine.set_xreg(0, 0x4000).unwrap();
        machine.set_xreg(1, 0x0400_0001_0000_0010).unwrap();
        machine.set_xreg(2, 0x80).unwrap();
        core.step_mte_word(C310_ADD_MOV_ALIGN_Y_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_SET_FLAG0_WORD, &fixture.source)
            .unwrap();
        core.step_mte_word(MTE2_TO_SCALAR_WAIT_FLAG0_WORD, &fixture.source)
            .unwrap();
        assert_eq!(core.ub().read_known(0, 128).unwrap(), x);
        assert_eq!(core.ub().read_known(0x80, 128).unwrap(), y);

        let mut vldi_xregs = [0_u64; 32];
        vldi_xregs[4] = 0;
        let mut rvec = C310RvecValueMachine::from_vector_words(vec![vec![0; 64]; 2]).unwrap();
        let step = rvec
            .execute_captured_vldi_word(
                0x10d0_db00,
                C310_CAPTURED_VLDI_V0_WORD,
                &vldi_xregs,
                core.ub(),
            )
            .unwrap();
        assert_eq!(&step.loaded_bytes[..128], x);
        assert_eq!(&step.loaded_bytes[128..], y);
    }

    #[test]
    fn unknown_mte_words_and_cross_architecture_words_do_not_advance() {
        let fixture = Fixture::new(&[0x5a; 32], false);
        for architecture in [Architecture::Dav2201, Architecture::Dav3510] {
            let mut core = stepper(architecture, 0x3000);
            let before = core.clone();
            assert!(matches!(
                core.step_mte_word(0x7000_0000, &fixture.source),
                Err(MteStepperError::UnsupportedWord {
                    pc: 0x1000,
                    word: 0x7000_0000,
                })
            ));
            assert_eq!(core, before);
            let wrong_arch_word = if architecture == Architecture::Dav2201 {
                C310_TILING_MOV_ALIGN_WORD
            } else {
                CAPTURED_C220_TILING_MOV_OUT_TO_UB_WORD
            };
            assert!(matches!(
                core.step_mte_word(wrong_arch_word, &fixture.source),
                Err(MteStepperError::UnsupportedWord { .. })
            ));
            assert_eq!(core, before);
        }
    }
}
