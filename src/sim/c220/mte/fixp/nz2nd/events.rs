use super::super::*;
use crate::memory::mapped::MappedMemory;
use crate::sim::c220::memory::{C220L0c, C220LocalBuffer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpNz2ndStage {
    GenerateRead,
    SendRead,
    SendL0c,
    ReceiveL0c,
    Convert,
    Slice,
    Transpose,
    Align,
    Packetize,
    GenerateWrite,
    SendWrite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpNz2ndEvent {
    Read(C220FixpEvent),
    Transposed(C220FixpTransposeProgress),
    Aligned(Option<C220FixpNz2ndStagingEntry>),
    Packetized(Option<C220FixpBiuWrite>),
    GeneratedWrite(C220FixpWriteProgress<C220FixpBiuWrite>),
    SentWrite(C220FixpWriteProgress<C220FixpBiuWrite>),
}

pub struct C220FixpNz2ndMemory<'a> {
    pub l0c: &'a mut C220L0c,
    pub slopes: &'a C220LocalBuffer,
    pub external: &'a mut MappedMemory,
    pub atomics: C220FixpAtomicConfig,
}

impl C220FixpNz2ndEngine {
    pub fn stage_ready_tick(&self, stage: C220FixpNz2ndStage, tick: u64) -> Option<u64> {
        use C220FixpNz2ndStage::*;
        match stage {
            GenerateRead => self.read_pipeline().generated_ready_tick(),
            SendRead => self
                .read_pipeline()
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
            SendL0c => self
                .read_interface()
                .input()
                .front()
                .map(|head| head.ready_tick),
            ReceiveL0c => self.read_interface().pending().front().map(|_| tick),
            Convert => self
                .read_interface()
                .acknowledgments()
                .front()
                .map(|head| head.ready_tick()),
            Slice => self
                .conversion()
                .entries()
                .front()
                .map(|head| head.ready_tick),
            Transpose => self.staging().transpose().ready_tick(),
            Align => self
                .staging()
                .alignment()
                .front()
                .map(|head| head.ready_tick),
            Packetize => self.output().bursts().front().map(|head| head.ready_tick),
            GenerateWrite => self
                .write_pipeline()
                .packets()
                .front()
                .map(|head| head.ready_tick),
            SendWrite => self
                .write_pipeline()
                .dispatch_queue()
                .front()
                .map(|head| head.ready_tick),
        }
    }
}
