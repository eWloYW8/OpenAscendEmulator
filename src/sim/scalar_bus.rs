use thiserror::Error;

use crate::architecture::Architecture;
use crate::isa::flow::{C310BufferStep, DcciStep, DsbStep, PipelineBarrierStep};
use crate::memory::hbm_pv_memory::{HbmPvMemory, HbmPvMemoryError};
use crate::memory::mapped::{MappedMemory, MappedMemoryError};
use crate::memory::sparse::MemoryByteState;
use crate::memory::ub::{UbMemory, UbMemoryError};
use crate::sim::c220::scalar_address::{
    C220_UB_BYTES, C220ScalarRoute, classify_c220_scalar_address,
};
use crate::sim::c310::buffer::C310BufferDisposition;
use crate::sim::c310::predicate_buffer::{C310PushPbDisposition, C310PushPbStep};
use crate::sim::c310::scalar_address::{
    C310_UB_ROUTE_BYTES, C310ScalarRoute, classify_c310_scalar_address,
};
use crate::sim::c310::vector_queue::{C310VfQueueDisposition, C310VfQueueStep};
use crate::sim::machine::ScalarMemoryBus;

impl ScalarMemoryBus for MappedMemory {
    type Error = MappedMemoryError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        let bytes = self.read_known_at(address, destination.len())?;
        destination.copy_from_slice(&bytes);
        Ok(())
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        self.write_known_at(address, source)
    }
}

impl ScalarMemoryBus for HbmPvMemory {
    type Error = HbmPvMemoryError;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        if destination.is_empty() {
            return Ok(());
        }
        self.device_to_host(address, destination)
    }

    fn write(&mut self, address: u64, source: &[u8]) -> Result<(), Self::Error> {
        if source.is_empty() {
            return Ok(());
        }
        self.host_to_device(address, source)
    }
}

#[derive(Debug, Error)]
pub enum UbScalarBusError<E: std::error::Error + 'static> {
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

pub(crate) struct UbScalarBus<'a, B> {
    ub: &'a mut UbMemory,
    fallback: &'a mut B,
    architecture: Architecture,
    local_roots: Option<(u64, u64)>,
}

enum ScalarBusRoute {
    Fallback(u64),
    Ub(u64),
}

impl<'a, B: ScalarMemoryBus> UbScalarBus<'a, B> {
    pub(crate) fn new(
        ub: &'a mut UbMemory,
        fallback: &'a mut B,
        architecture: Architecture,
        local_roots: Option<(u64, u64)>,
    ) -> Self {
        Self {
            ub,
            fallback,
            architecture,
            local_roots,
        }
    }

    #[cfg(test)]
    pub(crate) fn ub(&self) -> &UbMemory {
        self.ub
    }

    fn route(
        &self,
        address: u64,
        bytes: usize,
    ) -> Result<ScalarBusRoute, UbScalarBusError<B::Error>> {
        if bytes == 0 {
            return Ok(ScalarBusRoute::Fallback(address));
        }
        let length = u64::try_from(bytes).map_err(|_| UbScalarBusError::AddressOverflow)?;
        let end = address
            .checked_add(length)
            .ok_or(UbScalarBusError::AddressOverflow)?;
        let (spr67, spr68) = self
            .local_roots
            .ok_or(UbScalarBusError::MissingLocalRoots)?;
        match self.architecture {
            Architecture::Dav2201 => {
                let start = classify_c220_scalar_address(address, spr67, spr68);
                let last = classify_c220_scalar_address(end - 1, spr67, spr68);
                match (start, last) {
                    (C220ScalarRoute::Hbm, C220ScalarRoute::Hbm) => {
                        Ok(ScalarBusRoute::Fallback(address))
                    }
                    (C220ScalarRoute::Ub(offset), C220ScalarRoute::Ub(_))
                        if length <= C220_UB_BYTES.saturating_sub(offset) =>
                    {
                        Ok(ScalarBusRoute::Ub(offset))
                    }
                    (C220ScalarRoute::Unsupported, _) => {
                        Err(UbScalarBusError::UnsupportedLocal { address })
                    }
                    _ => Err(UbScalarBusError::AliasBoundary { address, bytes }),
                }
            }
            Architecture::Dav3510 => {
                let start = classify_c310_scalar_address(address, spr67, spr68);
                let last = classify_c310_scalar_address(end - 1, spr67, spr68);
                match (start, last) {
                    (C310ScalarRoute::Hbm(mapped), C310ScalarRoute::Hbm(last_mapped))
                        if mapped.checked_add(length - 1) == Some(last_mapped) =>
                    {
                        Ok(ScalarBusRoute::Fallback(mapped))
                    }
                    (C310ScalarRoute::Ub(offset), C310ScalarRoute::Ub(_))
                        if length <= C310_UB_ROUTE_BYTES.saturating_sub(offset) =>
                    {
                        Ok(ScalarBusRoute::Ub(offset))
                    }
                    (C310ScalarRoute::Unsupported, _) => {
                        Err(UbScalarBusError::UnsupportedLocal { address })
                    }
                    _ => Err(UbScalarBusError::AliasBoundary { address, bytes }),
                }
            }
        }
    }
}

impl<B: ScalarMemoryBus> ScalarMemoryBus for UbScalarBus<'_, B> {
    type Error = UbScalarBusError<B::Error>;

    fn read(&mut self, address: u64, destination: &mut [u8]) -> Result<(), Self::Error> {
        match self.route(address, destination.len())? {
            ScalarBusRoute::Ub(offset) => {
                destination.copy_from_slice(&self.ub.read_known(offset, destination.len())?);
                Ok(())
            }
            ScalarBusRoute::Fallback(mapped) => self
                .fallback
                .read(mapped, destination)
                .map_err(UbScalarBusError::Fallback),
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
                .map_err(UbScalarBusError::Fallback),
        }
    }

    fn maintain_data_cache(&mut self, step: DcciStep) -> Result<bool, Self::Error> {
        self.fallback
            .maintain_data_cache(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn synchronize_pipeline(&mut self, step: DsbStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_pipeline(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn synchronize_barrier(&mut self, step: PipelineBarrierStep) -> Result<bool, Self::Error> {
        self.fallback
            .synchronize_barrier(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn execute_c310_buffer(
        &mut self,
        step: C310BufferStep,
    ) -> Result<C310BufferDisposition, Self::Error> {
        self.fallback
            .execute_c310_buffer(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn execute_c310_push_pb(
        &mut self,
        step: C310PushPbStep,
    ) -> Result<C310PushPbDisposition, Self::Error> {
        self.fallback
            .execute_c310_push_pb(step)
            .map_err(UbScalarBusError::Fallback)
    }

    fn enqueue_c310_vf(
        &mut self,
        step: C310VfQueueStep,
    ) -> Result<C310VfQueueDisposition, Self::Error> {
        self.fallback
            .enqueue_c310_vf(step)
            .map_err(UbScalarBusError::Fallback)
    }
}
