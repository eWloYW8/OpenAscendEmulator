use std::num::NonZeroU32;

use super::{C220Load3dReadRequest, C220Load3dRequestError, C220Load3dV2Command};
use crate::isa::c220::mte::load3d::C220Load3dDestination;
use crate::sim::c220::memory::l1::C220L1Access;
use crate::sim::c220::mte::interface::{
    C220L0WritePort, C220MteL1OutputDestination, C220MteL1ReadOperation, C220MteL1ReadPort,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dReadUop {
    pub logical: C220Load3dReadRequest,
    pub source_address: u64,
    pub input_offset: u32,
    pub input_bytes: u32,
    pub completes_output: bool,
    pub last_in_instruction: bool,
}

impl C220Load3dReadUop {
    pub const INPUT_PORT: C220MteL1ReadPort = C220MteL1ReadPort::Port0;

    pub fn operation(
        self,
        instruction_id: u64,
        output_bandwidth: NonZeroU32,
    ) -> C220MteL1ReadOperation<Self> {
        C220MteL1ReadOperation {
            instruction_id,
            access: C220L1Access {
                address: self.source_address,
                bytes: self.input_bytes,
            },
            destination: match self.logical.destination {
                C220Load3dDestination::L0a => {
                    C220MteL1OutputDestination::L0a(C220L0WritePort::Port0)
                }
                C220Load3dDestination::L0b => {
                    C220MteL1OutputDestination::L0b(C220L0WritePort::Port0)
                }
            },
            output_address: u64::from(self.logical.destination_address),
            output_bytes: self.logical.output_bytes,
            output_bandwidth,
            completes_logical_uop: self.completes_output,
            last_in_instruction: self.last_in_instruction,
            payload: self,
        }
    }
}

#[derive(Debug, Clone)]
pub struct C220Load3dPhysicalReads {
    requests: std::vec::IntoIter<C220Load3dReadRequest>,
    current: Option<C220Load3dReadRequest>,
    offset: u32,
    access_width: NonZeroU32,
}

impl C220Load3dV2Command {
    pub fn physical_reads(
        self,
        access_width: NonZeroU32,
    ) -> Result<C220Load3dPhysicalReads, C220Load3dRequestError> {
        Ok(C220Load3dPhysicalReads {
            requests: self.read_requests()?.into_iter(),
            current: None,
            offset: 0,
            access_width,
        })
    }
}

impl Iterator for C220Load3dPhysicalReads {
    type Item = C220Load3dReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        let logical = loop {
            let logical = self.current.take().or_else(|| self.requests.next())?;
            if logical.input_bytes != 0 {
                break logical;
            }
        };
        let input_bytes = (logical.input_bytes - self.offset).min(self.access_width.get());
        let final_fragment = self.offset + input_bytes == logical.input_bytes;
        let uop = C220Load3dReadUop {
            logical,
            source_address: u64::from(logical.source_address) + u64::from(self.offset),
            input_offset: self.offset,
            input_bytes,
            completes_output: logical.completes_output && final_fragment,
            last_in_instruction: self.requests.len() == 0 && final_fragment,
        };
        if final_fragment {
            self.offset = 0;
        } else {
            self.current = Some(logical);
            self.offset += input_bytes;
        }
        Some(uop)
    }
}

impl std::iter::FusedIterator for C220Load3dPhysicalReads {}
