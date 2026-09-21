use crate::instruction::c220::mte::{
    C220DmaMovDescriptor, C220MovOutToUbDescriptor, C220MovOutToUbError, C220MovOutToUbSegment,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte2TransferPlan {
    pub descriptor: C220MovOutToUbDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}

impl C220Mte2TransferPlan {
    pub fn descriptor_segments(self) -> Result<Vec<C220MovOutToUbSegment>, C220MovOutToUbError> {
        self.descriptor
            .segments(self.source_address, self.destination_address)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Mte3TransferPlan {
    pub descriptor: C220DmaMovDescriptor,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: usize,
    pub dma_mode_word: u64,
}
