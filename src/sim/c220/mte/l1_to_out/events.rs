use super::C220L1OutputEngine;
use crate::sim::c220::mte::fixp::{
    C220FixpBiuWrite, C220FixpDispatchPacket, C220FixpReadProgress, C220FixpWriteProgress,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L1OutputStage {
    GenerateRead,
    SendRead,
    Packetize,
    GenerateWrite,
    SendWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L1OutputEvent {
    GeneratedRead(C220FixpReadProgress),
    SentRead(C220FixpReadProgress),
    SourceCompleted { tick: u64, instruction_id: u64 },
    Packetized(Option<C220FixpBiuWrite>),
    GeneratedWrite(C220FixpWriteProgress<C220FixpDispatchPacket>),
    SentWrite(C220FixpWriteProgress<C220FixpDispatchPacket>),
    WriteCompleted { tick: u64, instruction_id: u64 },
}

impl C220L1OutputEngine {
    pub fn stage_ready_tick(&self, stage: C220L1OutputStage) -> Option<u64> {
        match stage {
            C220L1OutputStage::GenerateRead => self.read_pipeline().generated_ready_tick(),
            C220L1OutputStage::SendRead => self
                .read_pipeline()
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
            C220L1OutputStage::Packetize => {
                self.output().bursts().front().map(|head| head.ready_tick)
            }
            C220L1OutputStage::GenerateWrite => self
                .write_pipeline()
                .packets()
                .front()
                .map(|head| head.ready_tick),
            C220L1OutputStage::SendWrite => self
                .write_pipeline()
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
        }
    }
}
