use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU32;

use super::C220BiuWriteDataReady;
use crate::sim::c220::mte::interface::biu_read::C220BiuSubcore;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteData {
    pub subcore: C220BiuSubcore,
    pub sent_tick: u64,
    pub ready_tick: u64,
    pub source: C220BiuWriteDataReady,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteDataSend {
    pub tick: u64,
    pub selected: Option<C220BiuSubcore>,
    pub sent: Option<C220BiuWriteData>,
    pub transport_full: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuWriteResponse {
    pub tick: u64,
    pub data: C220BiuWriteData,
}

impl C220BiuWriteResponse {
    pub fn retired_instruction(&self) -> Option<u64> {
        self.data
            .source
            .request
            .last_in_instruction
            .then_some(self.data.source.request.instruction_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220BiuWriteDataError {
    #[error("BIU write data time reversed from {previous} to {requested}")]
    TimeReversed { previous: u64, requested: u64 },
    #[error("BIU write data send already ran at tick {0}")]
    RepeatedSend(u64),
    #[error("BIU write data time overflowed")]
    TimeOverflow,
    #[error("BIU write tag {0} already has data in flight")]
    DuplicateTag(NonZeroU32),
    #[error("BIU write response tag {0} has no delivered data")]
    UnexpectedResponse(NonZeroU32),
}

/// Shared write-data port. Sending consumes transport credit, not a BIU tag;
/// final write responses and instruction retirement remain separate stages.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct C220BiuWriteDataPort {
    next: usize,
    transport: VecDeque<C220BiuWriteData>,
    delivered: BTreeMap<NonZeroU32, C220BiuWriteData>,
    observed_tick: Option<u64>,
    send_tick: Option<u64>,
}

impl C220BiuWriteDataPort {
    pub fn is_idle(&self) -> bool {
        self.transport.is_empty() && self.delivered.is_empty()
    }

    pub fn requests(&self) -> &VecDeque<C220BiuWriteData> {
        &self.transport
    }

    pub fn delivered(&self) -> impl Iterator<Item = &C220BiuWriteData> {
        self.delivered.values()
    }

    pub fn send(
        &mut self,
        tick: u64,
        heads: [Option<C220BiuWriteDataReady>; 3],
    ) -> Result<C220BiuWriteDataSend, C220BiuWriteDataError> {
        self.check_time(tick)?;
        if self.send_tick == Some(tick) {
            return Err(C220BiuWriteDataError::RepeatedSend(tick));
        }
        let selected = (0..3)
            .map(|offset| (self.next + offset) % 3)
            .find(|&index| heads[index].is_some_and(|head| head.ready_tick <= tick));
        let mut result = C220BiuWriteDataSend {
            tick,
            selected: None,
            sent: None,
            transport_full: false,
        };
        if let Some(index) = selected {
            let subcore = [
                C220BiuSubcore::Cube,
                C220BiuSubcore::Vector0,
                C220BiuSubcore::Vector1,
            ][index];
            result.selected = Some(subcore);
            result.transport_full = self.transport.len() == 2;
            if !result.transport_full {
                let tag = heads[index].expect("eligible source").request.tag;
                if self.delivered.contains_key(&tag)
                    || self
                        .transport
                        .iter()
                        .any(|data| data.source.request.tag == tag)
                {
                    return Err(C220BiuWriteDataError::DuplicateTag(tag));
                }
                let request = C220BiuWriteData {
                    subcore,
                    sent_tick: tick,
                    ready_tick: tick
                        .checked_add(1)
                        .ok_or(C220BiuWriteDataError::TimeOverflow)?,
                    source: heads[index].expect("eligible source"),
                };
                self.transport.push_back(request);
                result.sent = Some(request);
            }
            // Arbitration advances even when downstream credit is unavailable.
            self.next = (index + 1) % 3;
        }
        self.send_tick = Some(tick);
        self.observed_tick = Some(tick);
        Ok(result)
    }

    pub fn take_request(
        &mut self,
        tick: u64,
    ) -> Result<Option<C220BiuWriteData>, C220BiuWriteDataError> {
        self.check_time(tick)?;
        self.observed_tick = Some(tick);
        let request = self.transport.pop_front_if(|head| head.ready_tick <= tick);
        if let Some(data) = request {
            self.delivered.insert(data.source.request.tag, data);
        }
        Ok(request)
    }

    pub fn receive_response(
        &mut self,
        tick: u64,
        tag: NonZeroU32,
    ) -> Result<C220BiuWriteResponse, C220BiuWriteDataError> {
        self.check_time(tick)?;
        let data = self
            .delivered
            .remove(&tag)
            .ok_or(C220BiuWriteDataError::UnexpectedResponse(tag))?;
        self.observed_tick = Some(tick);
        Ok(C220BiuWriteResponse { tick, data })
    }

    fn check_time(&self, tick: u64) -> Result<(), C220BiuWriteDataError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220BiuWriteDataError::TimeReversed {
                previous,
                requested: tick,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::c220::mte::interface::biu_write::C220BiuWriteSourceRequest;

    #[test]
    fn backpressure_retains_heads_but_advances_round_robin() {
        let heads = std::array::from_fn(|index| {
            Some(C220BiuWriteDataReady {
                ready_tick: 0,
                request: C220BiuWriteSourceRequest {
                    tag: std::num::NonZeroU32::new(index as u32 + 1).unwrap(),
                    instruction_id: index as u64,
                    source_address: 0,
                    bytes: 32,
                    gather_stride: None,
                    last_in_instruction: true,
                },
            })
        });
        let mut port = C220BiuWriteDataPort::default();
        assert_eq!(
            port.send(0, heads).unwrap().sent.unwrap().subcore,
            C220BiuSubcore::Cube
        );
        assert!(port.take_request(0).unwrap().is_none());
        assert_eq!(
            port.send(1, heads).unwrap().sent.unwrap().subcore,
            C220BiuSubcore::Vector0
        );
        let stalled = port.send(2, heads).unwrap();
        assert_eq!(stalled.selected, Some(C220BiuSubcore::Vector1));
        assert!(stalled.transport_full && stalled.sent.is_none());
        assert_eq!(port.requests().len(), 2);
        assert_eq!(
            port.take_request(2).unwrap().unwrap().subcore,
            C220BiuSubcore::Cube
        );
        let tag = heads[0].unwrap().request.tag;
        assert!(matches!(
            port.send(3, heads),
            Err(C220BiuWriteDataError::DuplicateTag(_))
        ));
        let response = port.receive_response(3, tag).unwrap();
        assert_eq!(response.retired_instruction(), Some(0));
        assert!(port.receive_response(3, tag).is_err());
        assert_eq!(
            port.send(3, heads).unwrap().sent.unwrap().subcore,
            C220BiuSubcore::Cube
        );
        assert!(matches!(
            port.send(3, heads),
            Err(C220BiuWriteDataError::RepeatedSend(3))
        ));
        assert_eq!(port.requests().len(), 2);
    }
}
