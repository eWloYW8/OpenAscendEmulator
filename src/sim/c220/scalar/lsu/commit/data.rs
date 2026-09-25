use super::*;
use crate::sim::c220::scalar::C220ScalarMappedAddress;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LoadResponse {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub mapped: C220ScalarMappedAddress,
    pub path: C220LsuLoadPath,
    pub part: C220LsuPairPart,
}

impl From<C220LsuLoadValue> for C220LoadResponse {
    fn from(data: C220LsuLoadValue) -> Self {
        Self {
            request: data.request,
            tick: data.tick,
            mapped: data.mapped,
            path: data.path,
            part: data.part,
        }
    }
}

impl C220LsuCommitLane {
    /// Data becomes available now; architectural retirement remains separate.
    /// A rejected completion remains owned by the caller, without register changes.
    pub fn complete_data_at(
        &mut self,
        tick: u64,
        data: C220LsuLoadValue,
        machine: &mut ScalarMachine,
    ) -> Result<(), C220LsuCommitError> {
        self.check_tick(tick)?;
        Self::check_machine(machine, data.operands)?;
        let ready = tick.checked_add(1).ok_or(C220LsuCommitError::Overflow)?;
        let instruction = *self
            .requests
            .get(&data.request)
            .ok_or(C220LsuCommitError::InvalidRequest)?;
        let pending = *self
            .pending
            .get(&instruction)
            .ok_or(C220LsuCommitError::InvalidRequest)?;
        let index = match (pending.second_request, data.part) {
            (None, C220LsuPairPart::Both) => 0,
            (Some(second), C220LsuPairPart::First) if data.request != second => 0,
            (Some(second), C220LsuPairPart::Second) if data.request == second => 1,
            _ => return Err(C220LsuCommitError::InvalidRequest),
        };
        if pending.operands != data.operands
            || data.second_value.is_some() != data.operands.second_destination.is_some()
            || pending.responses[index].is_some()
            || data.tick > tick
            || data.tick < pending.admission.expect("bound request").0
        {
            return Err(C220LsuCommitError::InvalidRequest);
        }
        let mut responses = pending.responses;
        responses[index] = Some(data);
        let complete = pending.second_request.is_none() || responses.iter().all(Option::is_some);
        let notify = complete
            || !matches!(
                data.path,
                C220LsuLoadPath::Cache | C220LsuLoadPath::CacheAndStore
            );
        if notify && self.retirements.len() == 64 {
            return Err(C220LsuCommitError::RetirementFull);
        }
        let combined = if pending.second_request.is_some() {
            C220LsuLoadValue {
                request: pending.admission.expect("bound request").1,
                mapped: responses[0].map_or_else(
                    || C220ScalarMappedAddress {
                        address: data
                            .mapped
                            .address
                            .wrapping_sub(u64::from(data.operands.width_bytes)),
                        ..data.mapped
                    },
                    |first| first.mapped,
                ),
                value: responses[0].map_or(0, |first| first.value),
                second_value: Some(
                    responses[1]
                        .and_then(|second| second.second_value)
                        .unwrap_or(0),
                ),
                part: C220LsuPairPart::Both,
                ..data
            }
        } else {
            data
        };
        if self.mode == C220LoadCommitMode::DataBypass && !pending.suppressed {
            self.write_result(combined, machine);
            self.pending
                .get_mut(&instruction)
                .expect("checked load")
                .writeback_tick = Some(tick);
        }
        let pending = self.pending.get_mut(&instruction).expect("checked load");
        pending.responses = responses;
        pending.data = complete.then_some(combined);
        if notify {
            self.retirements
                .push_back((ready, RetirementToken::Load(instruction)));
        }
        self.tick = tick;
        Ok(())
    }
}
