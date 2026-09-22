use thiserror::Error;

use crate::isa::flow::{DcciStep, DsbStep, PipelineBarrierStep};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::sim::c220::scalar::address::{
    C220_UB_BYTES, C220ScalarRoute, classify_c220_scalar_address,
};
use crate::sim::common::scalar::ScalarMemoryBus;

#[derive(Debug, Error)]
pub enum C220ScalarBusError<E: std::error::Error + 'static> {
    #[error("scalar address range overflows u64")]
    AddressOverflow,
    #[error(
        "scalar address range at {address:#x} with {bytes} bytes crosses the UB alias boundary"
    )]
    AliasBoundary { address: u64, bytes: usize },
    #[error("scalar local-address roots are unavailable")]
    MissingLocalRoots,
    #[error("scalar local address {address:#x} is unsupported")]
    UnsupportedLocal { address: u64 },
    #[error(transparent)]
    Ub(#[from] UbMemoryError),
    #[error("scalar memory backend: {0}")]
    Fallback(#[source] E),
}

pub(crate) struct C220ScalarBus<'a, B> {
    ub: &'a mut UbMemory,
    fallback: &'a mut B,
    local_roots: Option<(u64, u64)>,
}

enum ScalarBusRoute {
    Fallback(u64),
    Ub(u64),
}

impl<'a, B: ScalarMemoryBus> C220ScalarBus<'a, B> {
    pub(crate) fn new(
        ub: &'a mut UbMemory,
        fallback: &'a mut B,
        local_roots: Option<(u64, u64)>,
    ) -> Self {
        Self {
            ub,
            fallback,
            local_roots,
        }
    }

    fn route(
        &self,
        address: u64,
        bytes: usize,
    ) -> Result<ScalarBusRoute, C220ScalarBusError<B::Error>> {
        if bytes == 0 {
            return Ok(ScalarBusRoute::Fallback(address));
        }
        let length = u64::try_from(bytes).map_err(|_| C220ScalarBusError::AddressOverflow)?;
        let end = address
            .checked_add(length)
            .ok_or(C220ScalarBusError::AddressOverflow)?;
        let (spr67, spr68) = self
            .local_roots
            .ok_or(C220ScalarBusError::MissingLocalRoots)?;
        let start = classify_c220_scalar_address(address, spr67, spr68);
        let last = classify_c220_scalar_address(end - 1, spr67, spr68);
        match (start, last) {
            (C220ScalarRoute::Hbm, C220ScalarRoute::Hbm) => Ok(ScalarBusRoute::Fallback(address)),
            (C220ScalarRoute::Ub(offset), C220ScalarRoute::Ub(_))
                if length <= C220_UB_BYTES.saturating_sub(offset) =>
            {
                Ok(ScalarBusRoute::Ub(offset))
            }
            (C220ScalarRoute::Unsupported, _) => {
                Err(C220ScalarBusError::UnsupportedLocal { address })
            }
            _ => Err(C220ScalarBusError::AliasBoundary { address, bytes }),
        }
    }
}

impl<B: ScalarMemoryBus> ScalarMemoryBus for C220ScalarBus<'_, B> {
    type Error = C220ScalarBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        match self.route(address, destination.len())? {
            ScalarBusRoute::Ub(offset) => {
                destination.copy_from_slice(&self.ub.read_known(offset, destination.len())?);
                Ok(())
            }
            ScalarBusRoute::Fallback(mapped) => self
                .fallback
                .read(mapped, destination)
                .map_err(C220ScalarBusError::Fallback),
        }
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        match self.route(address, source.len())? {
            ScalarBusRoute::Ub(offset) => {
                let states = source
                    .iter()
                    .copied()
                    .map(MemoryByteState::Known)
                    .collect::<Vec<_>>();
                self.ub.write_states(offset, &states)?;
                Ok(())
            }
            ScalarBusRoute::Fallback(mapped) => self
                .fallback
                .write(mapped, source)
                .map_err(C220ScalarBusError::Fallback),
        }
    }

    fn maintain_data_cache(&mut self, step: DcciStep) -> Result<bool, Self::Error> {
        self.fallback
            .maintain_data_cache(step)
            .map_err(C220ScalarBusError::Fallback)
    }

    fn synchronize_pipeline(&mut self, step: DsbStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_pipeline(step)
            .map_err(C220ScalarBusError::Fallback)
    }

    fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_barrier(step)
            .map_err(C220ScalarBusError::Fallback)
    }
}
