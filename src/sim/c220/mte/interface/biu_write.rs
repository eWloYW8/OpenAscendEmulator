use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::ub_read::{C220UbReadError, C220UbReadFragment, C220UbReadInterface};

pub mod command;
pub mod data;

/// Source metadata for one issued BIU write transaction, after BIU splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteSourceRequest {
    pub tag: NonZeroU32,
    pub instruction_id: u64,
    pub source_address: u64,
    pub bytes: u32,
    pub gather_stride: Option<u32>,
    pub last_in_instruction: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteDataReady {
    pub ready_tick: u64,
    pub request: C220BiuWriteSourceRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuWriteSourceEvent {
    ReadPacket(C220UbReadFragment),
    ReadComplete { tag: NonZeroU32, remaining: u32 },
    DataReady(C220BiuWriteDataReady),
}

#[derive(Debug, thiserror::Error)]
pub enum C220BiuWriteSourceError {
    #[error("BIU write tag {0} is already active")]
    DuplicateTag(NonZeroU32),
    #[error("BIU write tag {0} has no pending command response")]
    UnexpectedDbid(NonZeroU32),
    #[error("BIU write source time overflowed")]
    TimeOverflow,
    #[error(transparent)]
    UbRead(#[from] C220UbReadError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SourcePackets {
    tag: NonZeroU32,
    address: u64,
    remaining_bytes: u32,
    remaining_packets: u32,
    width: u32,
    stride: Option<u32>,
}

impl SourcePackets {
    fn new(request: C220BiuWriteSourceRequest, bandwidth: NonZeroU32, gather_offset: u32) -> Self {
        if let Some(stride) = request.gather_stride {
            let count = request.bytes >> 6;
            Self {
                tag: request.tag,
                address: request
                    .source_address
                    .wrapping_add(u64::from(gather_offset)),
                remaining_bytes: count * 64,
                remaining_packets: count,
                width: 64,
                stride: Some(stride),
            }
        } else {
            let address = (request.source_address as u32) & !31;
            let bytes = request
                .bytes
                .wrapping_add(request.source_address as u32)
                .wrapping_sub(address);
            Self {
                tag: request.tag,
                address: u64::from(address),
                remaining_bytes: bytes,
                remaining_packets: bytes.div_ceil(bandwidth.get()),
                width: bandwidth.get(),
                stride: None,
            }
        }
    }

    fn front(&self) -> Option<C220UbReadFragment> {
        (self.remaining_packets != 0).then(|| C220UbReadFragment {
            tag: self.tag.get(),
            address: self.address,
            bytes: self.remaining_bytes.min(self.width),
            completes_read: true,
        })
    }

    fn pop(&mut self) {
        let bytes = self.remaining_bytes.min(self.width);
        self.remaining_bytes -= bytes;
        self.remaining_packets -= 1;
        self.address = self
            .address
            .wrapping_add(u64::from(self.stride.unwrap_or(bytes)));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PendingRead {
    request: C220BiuWriteSourceRequest,
    dbid_received: bool,
    remaining_responses: u32,
    data_sent: bool,
}

/// DBID-gated source side of one vector subcore's BIU write interface.
/// The command issuer owns tags until final write response; this component
/// only produces data-ready transactions, never instruction completions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220BiuWriteSource {
    bandwidth: NonZeroU32,
    pending: BTreeMap<NonZeroU32, PendingRead>,
    ingress: VecDeque<(u64, NonZeroU32)>,
    egress: VecDeque<(u64, NonZeroU32)>,
    data_ready: VecDeque<C220BiuWriteDataReady>,
    packets: Option<SourcePackets>,
    gather_offset: u32,
}

impl C220BiuWriteSource {
    pub fn new(bandwidth: NonZeroU32) -> Self {
        Self {
            bandwidth,
            pending: BTreeMap::new(),
            ingress: VecDeque::new(),
            egress: VecDeque::new(),
            data_ready: VecDeque::new(),
            packets: None,
            gather_offset: 0,
        }
    }

    pub fn is_idle(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn data_ready(&self) -> &VecDeque<C220BiuWriteDataReady> {
        &self.data_ready
    }
    pub fn pending_requests(&self) -> impl Iterator<Item = C220BiuWriteSourceRequest> + '_ {
        self.pending.values().map(|pending| pending.request)
    }

    pub fn register(
        &mut self,
        request: C220BiuWriteSourceRequest,
    ) -> Result<(), C220BiuWriteSourceError> {
        if self.pending.contains_key(&request.tag) {
            return Err(C220BiuWriteSourceError::DuplicateTag(request.tag));
        }
        self.pending.insert(
            request.tag,
            PendingRead {
                request,
                dbid_received: false,
                remaining_responses: 0,
                data_sent: false,
            },
        );
        Ok(())
    }

    pub fn receive_dbid(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<(), C220BiuWriteSourceError> {
        let pending = self
            .pending
            .get_mut(&tag)
            .filter(|pending| !pending.dbid_received)
            .ok_or(C220BiuWriteSourceError::UnexpectedDbid(tag))?;
        let ready_tick = tick
            .checked_add(1)
            .ok_or(C220BiuWriteSourceError::TimeOverflow)?;
        pending.dbid_received = true;
        self.ingress.push_back((ready_tick, tag));
        Ok(())
    }

    pub(in crate::sim::c220::mte) fn advance(
        &mut self,
        tick: u64,
        ub: &mut C220UbReadInterface,
    ) -> Result<Vec<C220BiuWriteSourceEvent>, C220BiuWriteSourceError> {
        let mut events = Vec::new();
        if let Some(&(ready, tag)) = self.ingress.front()
            && ready <= tick
        {
            // Check the next stage's deadline before changing packet ownership.
            let next_tick = tick
                .checked_add(1)
                .ok_or(C220BiuWriteSourceError::TimeOverflow)?;
            if self.packets.is_none() {
                let pending = self.pending.get_mut(&tag).expect("registered ingress");
                let packets =
                    SourcePackets::new(pending.request, self.bandwidth, self.gather_offset);
                pending.remaining_responses = packets.remaining_packets;
                if let Some(stride) = pending.request.gather_stride {
                    self.gather_offset = self
                        .gather_offset
                        .wrapping_add(stride.wrapping_mul(packets.remaining_packets));
                    if pending.request.last_in_instruction {
                        self.gather_offset = 0;
                    }
                }
                self.packets = Some(packets);
            }
            let packets = self.packets.as_mut().expect("active ingress packets");
            if let Some(fragment) = packets.front()
                && ub.push(tick, fragment)?
            {
                packets.pop();
                events.push(C220BiuWriteSourceEvent::ReadPacket(fragment));
            }
            if packets.front().is_none() {
                self.packets = None;
                self.ingress.pop_front();
                self.egress.push_back((next_tick, tag));
            }
        }
        if let Some(&(ready, tag)) = self.egress.front()
            && ready <= tick
        {
            let next_tick = tick
                .checked_add(1)
                .ok_or(C220BiuWriteSourceError::TimeOverflow)?;
            if ub.take_completion(tick, tag.get())?.is_some() {
                let pending = self.pending.get_mut(&tag).expect("registered egress");
                pending.remaining_responses -= 1;
                events.push(C220BiuWriteSourceEvent::ReadComplete {
                    tag,
                    remaining: pending.remaining_responses,
                });
                if pending.remaining_responses == 0 {
                    let data = C220BiuWriteDataReady {
                        ready_tick: next_tick,
                        request: pending.request,
                    };
                    self.egress.pop_front();
                    self.data_ready.push_back(data);
                    events.push(C220BiuWriteSourceEvent::DataReady(data));
                }
            }
        }
        Ok(events)
    }

    pub(in crate::sim::c220::mte) fn starting_source(&self, tick: u64) -> Option<NonZeroU32> {
        self.ingress
            .front()
            .filter(|(ready, _)| *ready <= tick && self.packets.is_none())
            .map(|(_, tag)| *tag)
    }

    pub(in crate::sim::c220::mte) fn set_source_tail(&mut self, tag: NonZeroU32, tail: bool) {
        self.pending
            .get_mut(&tag)
            .expect("registered ingress")
            .request
            .last_in_instruction = tail;
    }

    pub fn take_data_ready(&mut self, tick: u64) -> Option<C220BiuWriteDataReady> {
        let data = self
            .data_ready
            .pop_front_if(|head| head.ready_tick <= tick)?;
        self.pending
            .get_mut(&data.request.tag)
            .expect("registered data")
            .data_sent = true;
        Some(data)
    }

    pub(in crate::sim::c220::mte) fn release_response(&mut self, tag: NonZeroU32) {
        let pending = self
            .pending
            .remove(&tag)
            .expect("delivered data retains its tag");
        assert!(pending.data_sent);
    }
}
