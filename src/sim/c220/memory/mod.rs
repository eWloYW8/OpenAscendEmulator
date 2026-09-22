mod buffer;
mod l0c;

use crate::architecture::c220::C220Device;

pub use buffer::{C220LocalBuffer, C220LocalBufferError};
pub use l0c::{
    C220_L0C_FRAGMENT_BYTES, C220L0c, C220L0cError, C220L0cFragmentRequest, C220L0cMaster,
    C220L0cScoreboard, C220L0cUnitFlagBlock, C220L0cWriteArbiter,
};

pub const C220_L0C_UNIT_FLAG_READ_LATENCY: u32 = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LocalMemoryConfig {
    pub l0a_bytes: u64,
    pub l0b_bytes: u64,
    pub l0c_bytes: u64,
    pub l1_bytes: u64,
    pub l0c_unit_flag_read_latency: u32,
}

impl C220LocalMemoryConfig {
    pub const fn for_device(device: C220Device) -> Self {
        let profile = device.profile();
        Self {
            l0a_bytes: profile.l0a_bytes as u64,
            l0b_bytes: profile.l0b_bytes as u64,
            l0c_bytes: profile.l0c_bytes as u64,
            l1_bytes: profile.l1_bytes as u64,
            l0c_unit_flag_read_latency: C220_L0C_UNIT_FLAG_READ_LATENCY,
        }
    }
}

impl Default for C220LocalMemoryConfig {
    fn default() -> Self {
        Self::for_device(C220Device::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220LocalMemory {
    l0a: C220LocalBuffer,
    l0b: C220LocalBuffer,
    l0c: C220L0c,
    l1: C220LocalBuffer,
}

impl C220LocalMemory {
    pub fn new(config: C220LocalMemoryConfig) -> Result<Self, C220L0cError> {
        Ok(Self {
            l0a: C220LocalBuffer::new(config.l0a_bytes),
            l0b: C220LocalBuffer::new(config.l0b_bytes),
            l0c: C220L0c::new(config.l0c_bytes, config.l0c_unit_flag_read_latency)?,
            l1: C220LocalBuffer::new(config.l1_bytes),
        })
    }

    pub const fn l0a(&self) -> &C220LocalBuffer {
        &self.l0a
    }

    pub fn l0a_mut(&mut self) -> &mut C220LocalBuffer {
        &mut self.l0a
    }

    pub const fn l0b(&self) -> &C220LocalBuffer {
        &self.l0b
    }

    pub fn l0b_mut(&mut self) -> &mut C220LocalBuffer {
        &mut self.l0b
    }

    pub const fn l0c(&self) -> &C220L0c {
        &self.l0c
    }

    pub fn l0c_mut(&mut self) -> &mut C220L0c {
        &mut self.l0c
    }

    pub const fn l1(&self) -> &C220LocalBuffer {
        &self.l1
    }

    pub fn l1_mut(&mut self) -> &mut C220LocalBuffer {
        &mut self.l1
    }
}
