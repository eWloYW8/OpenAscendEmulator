use super::*;
use crate::sim::c220::scalar::lsu::scheduler::C220LsuStorePath;
use crate::sim::c220::scalar::{C220ScalarMappedAddress, C220StoreOperands};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220StoreResponse {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub mapped: C220ScalarMappedAddress,
    pub path: C220LsuStorePath,
    pub part: C220LsuPairPart,
}

impl From<C220LsuStoreValue> for C220StoreResponse {
    fn from(data: C220LsuStoreValue) -> Self {
        Self {
            request: data.request,
            tick: data.tick,
            mapped: data.mapped,
            path: data.path,
            part: data.part,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingStorePair {
    requests: [C220LsuRequestId; 2],
    operands: C220StoreOperands,
    admission_tick: u64,
    responses: [Option<C220StoreResponse>; 2],
    consumed: u8,
    retired: bool,
}

impl C220LsuCommitLane {
    pub fn admit_store_pair(
        &mut self,
        tick: u64,
        first: C220LsuRequestId,
        second: C220LsuRequestId,
        operands: C220StoreOperands,
    ) -> Result<(), C220LsuCommitError> {
        self.check_tick(tick)?;
        self.check_store_request(first)?;
        self.check_store_request(second)?;
        if first == second || operands.second_source_operand.is_none() {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        self.store_pairs.insert(
            first,
            PendingStorePair {
                requests: [first, second],
                operands,
                admission_tick: tick,
                responses: [None; 2],
                consumed: 0,
                retired: false,
            },
        );
        self.store_pair_requests.insert(first, first);
        self.store_pair_requests.insert(second, first);
        self.tick = tick;
        Ok(())
    }

    pub fn complete_store_at(
        &mut self,
        tick: u64,
        data: C220LsuStoreValue,
    ) -> Result<(), C220LsuCommitError> {
        self.check_retirement_send(tick)?;
        if data.tick > tick {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        if let Some(first) = self.store_pair_requests.get(&data.request) {
            let pair = self.store_pairs.get_mut(first).expect("bound store pair");
            let index = usize::from(data.request == pair.requests[1]);
            let part = if index == 0 {
                C220LsuPairPart::First
            } else {
                C220LsuPairPart::Second
            };
            if pair.responses[index].is_some()
                || data.operands != pair.operands
                || data.part != part
                || data.tick < pair.admission_tick
            {
                return Err(C220LsuCommitError::InvalidRequest);
            }
            pair.responses[index] = Some(data.into());
        } else {
            self.check_store_request(data.request)?;
            if data.part != C220LsuPairPart::Both {
                return Err(C220LsuCommitError::InvalidRequest);
            }
        }
        self.retirements
            .push_back((tick + 1, RetirementToken::Store(data)));
        self.tick = tick;
        Ok(())
    }

    pub(super) fn retire_store(
        &mut self,
        tick: u64,
        data: C220LsuStoreValue,
    ) -> Option<C220LsuRetirement> {
        let mut first_request = data.request;
        let mut repeated_notification = false;
        let mut final_notification = true;
        let mut responses = [Some(data.into()), None];
        if let Some(first) = self.store_pair_requests.get(&data.request).copied() {
            first_request = first;
            let pair = self.store_pairs.get_mut(&first).expect("bound store pair");
            pair.consumed += 1;
            if !pair.responses.iter().all(Option::is_some) {
                return None;
            }
            responses = pair.responses;
            repeated_notification = pair.retired;
            pair.retired = true;
            final_notification = pair.consumed == 2;
            if final_notification {
                let pair = self
                    .store_pairs
                    .remove(&first)
                    .expect("completed store pair");
                for request in pair.requests {
                    self.store_pair_requests.remove(&request);
                }
            }
        }
        Some(C220LsuRetirement::Store {
            data,
            retire_tick: tick,
            first_request,
            final_notification,
            repeated_notification,
            responses,
        })
    }
}
