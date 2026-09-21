use thiserror::Error;

use crate::execution::c220::ub_arbiter::{
    C220UbDecision, C220UbPort, C220UbRequest, C220UbRequestError,
};
use crate::instruction::c220::vector::{
    C220_VECTOR_BLOCK_BYTES, C220_VECTOR_TILE_BYTES, C220Fp32Addresses, C220Fp32Control,
    C220Fp32Issue, C220Fp32ReadAccess, C220VecArithmeticHint, C220VectorError, C220VectorStore,
    evaluate_c220_fp32_repeat_from_bytes,
};
use crate::instruction::fp32_vector::Fp32LaneOutcome;
use crate::memory::ub::UbMemory;

#[derive(Debug, Error)]
pub enum C220VectorReadError {
    #[error("FP32 repeat has no iteration mask")]
    MissingRepeatMask,
    #[error(transparent)]
    ReadPlan(#[from] C220VectorError),
    #[error(transparent)]
    UbRequest(#[from] C220UbRequestError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PendingFp32Read {
    pc: u64,
    word: u32,
    hint: C220VecArithmeticHint,
    control: C220Fp32Control,
    addresses: C220Fp32Addresses,
    mask: [u64; 4],
    repeat_index: usize,
    accesses: Vec<C220Fp32ReadAccess>,
    port0: ReadPort,
    port1: ReadPort,
    source_0_bytes: Vec<u8>,
    source_1_bytes: Vec<u8>,
    ready_tick: Option<u64>,
    sampled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReadPort {
    request: C220UbRequest,
    destination_offsets: Vec<usize>,
}

impl ReadPort {
    fn new(accesses: &[C220Fp32ReadAccess]) -> Result<Self, C220UbRequestError> {
        let spans = accesses
            .iter()
            .map(|access| (access.address, C220_VECTOR_BLOCK_BYTES))
            .collect::<Vec<_>>();
        let request = C220UbRequest::from_accesses(&spans)?;
        let mut destination_offsets = Vec::with_capacity(request.blocks().len());
        let mut next_block = 0;
        for access in accesses {
            let mut address = access.address;
            let end = address + C220_VECTOR_BLOCK_BYTES as u64;
            while address < end {
                let block = request.blocks()[next_block];
                debug_assert_eq!(block.address, address);
                destination_offsets.push(
                    usize::from(access.block_index) * C220_VECTOR_BLOCK_BYTES
                        + (address - access.address) as usize,
                );
                address += u64::from(block.bytes);
                next_block += 1;
            }
        }
        Ok(Self {
            request,
            destination_offsets,
        })
    }
}

impl PendingFp32Read {
    pub(super) fn new(
        issue: &C220Fp32Issue,
        repeat_index: usize,
    ) -> Result<Self, C220VectorReadError> {
        let mask = *issue
            .iteration_masks
            .get(repeat_index)
            .ok_or(C220VectorReadError::MissingRepeatMask)?;
        let accesses = issue.read_accesses_for_repeat(repeat_index)?;
        let port0_accesses = accesses
            .iter()
            .copied()
            .filter(|access| access.source_index == 0)
            .collect::<Vec<_>>();
        let port1_accesses = accesses
            .iter()
            .copied()
            .filter(|access| access.source_index == 1)
            .collect::<Vec<_>>();
        Ok(Self {
            pc: issue.pc,
            word: issue.word,
            hint: issue.hint,
            control: issue.control,
            addresses: issue.addresses,
            mask,
            repeat_index,
            accesses,
            port0: ReadPort::new(&port0_accesses)?,
            port1: ReadPort::new(&port1_accesses)?,
            source_0_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            source_1_bytes: vec![0; C220_VECTOR_TILE_BYTES],
            ready_tick: None,
            sampled: false,
        })
    }

    fn port(&self, port: C220UbPort) -> &ReadPort {
        match port {
            C220UbPort::VectorRead0 => &self.port0,
            C220UbPort::VectorRead1 => &self.port1,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        }
    }

    fn port_mut(&mut self, port: C220UbPort) -> &mut ReadPort {
        match port {
            C220UbPort::VectorRead0 => &mut self.port0,
            C220UbPort::VectorRead1 => &mut self.port1,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        }
    }

    pub(super) fn request(&self, port: C220UbPort) -> &C220UbRequest {
        &self.port(port).request
    }

    pub(super) fn request_mut(&mut self, port: C220UbPort) -> &mut C220UbRequest {
        &mut self.port_mut(port).request
    }

    pub(super) fn ready_tick(&self) -> Option<u64> {
        self.ready_tick
    }

    pub(super) fn set_ready_tick(&mut self, tick: u64) {
        self.ready_tick = Some(tick);
    }

    pub(super) fn is_sampled(&self) -> bool {
        self.sampled
    }

    pub(super) fn mark_sampled(&mut self) {
        self.sampled = true;
    }

    pub(super) fn capture(
        &mut self,
        decision: &C220UbDecision,
        ub: &UbMemory,
    ) -> Result<(), C220VectorError> {
        let offset = self.port(decision.port).destination_offsets[decision.block_index];
        let destination = match decision.port {
            C220UbPort::VectorRead0 => &mut self.source_0_bytes,
            C220UbPort::VectorRead1 => &mut self.source_1_bytes,
            C220UbPort::VectorWrite => unreachable!("read request selected a write port"),
        };
        let bytes = ub.read_known(decision.block.address, usize::from(decision.block.bytes))?;
        destination[offset..offset + bytes.len()].copy_from_slice(&bytes);
        Ok(())
    }

    pub(super) fn grant_tick(&self, admission_tick: u64) -> Option<u64> {
        if !self.port0.request.is_complete() || !self.port1.request.is_complete() {
            return None;
        }
        Some(
            self.port0
                .request
                .completion_tick()
                .into_iter()
                .chain(self.port1.request.completion_tick())
                .max()
                .unwrap_or(admission_tick),
        )
    }

    pub(super) fn sample(
        &self,
        ub: &UbMemory,
    ) -> Result<(C220Fp32ReadSample, Vec<C220VectorStore>), C220VectorError> {
        let result = evaluate_c220_fp32_repeat_from_bytes(
            self.hint,
            self.control,
            self.addresses,
            self.repeat_index,
            &self.mask,
            self.source_0_bytes.clone(),
            self.source_1_bytes.clone(),
            ub,
        )?;
        Ok((
            C220Fp32ReadSample {
                pc: self.pc,
                word: self.word,
                repeat_index: self.repeat_index,
                tick: self.ready_tick.expect("read sample has a ready tick"),
                accesses: self.accesses.clone(),
                read0_grants: self.port0.request.grants().to_vec(),
                read1_grants: self.port1.request.grants().to_vec(),
                source_0_bytes: result.source_0_bytes,
                source_1_bytes: result.source_1_bytes,
                lanes: result.lanes,
            },
            result.stores,
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220Fp32ReadSample {
    pub pc: u64,
    pub word: u32,
    pub repeat_index: usize,
    pub tick: u64,
    pub accesses: Vec<C220Fp32ReadAccess>,
    pub read0_grants: Vec<Option<u64>>,
    pub read1_grants: Vec<Option<u64>>,
    pub source_0_bytes: Vec<u8>,
    pub source_1_bytes: Vec<u8>,
    pub lanes: Vec<Fp32LaneOutcome>,
}
