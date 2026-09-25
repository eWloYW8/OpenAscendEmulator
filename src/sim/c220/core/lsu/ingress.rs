use super::*;
use crate::sim::c220::scalar::lsu::commit::C220LoadId;
use crate::sim::c220::scalar::lsu::store_buffer::C220LsuMemory;
use crate::sim::common::scalar::ScalarMachine;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreLsuAdmission {
    pub instruction_id: u64,
    pub request: C220LsuRequestId,
    pub second_request: Option<C220LsuRequestId>,
    pub tick: u64,
    pub mapped: C220ScalarMappedAddress,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DispatchedLsu {
    Load(C220CoreLoadIssue),
    DirectStore(C220CoreLsuIssue),
    Store(C220CoreStoreIssue),
}

impl DispatchedLsu {
    fn issue_tick(self) -> u64 {
        match self {
            Self::Load(issue) => issue.tick,
            Self::Store(issue) => issue.tick,
            Self::DirectStore(issue) => issue.tick,
        }
    }
}

impl C220Core {
    pub fn lsu_ingress_occupancy(&self) -> usize {
        self.lsu.as_ref().map_or(0, |lsu| lsu.ingress.len())
    }

    pub fn take_lsu_admissions(&mut self) -> Vec<C220CoreLsuAdmission> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.admissions))
            .unwrap_or_default()
    }
}

impl CoreLsu {
    /// Transfer one aged head. Cache backpressure retains its captured operands.
    pub(super) fn admit_ingress_at(
        &mut self,
        tick: u64,
        machine: &ScalarMachine,
    ) -> Result<(), C220CoreError> {
        let Some(head) = self
            .ingress
            .front()
            .copied()
            .filter(|head| head.issue_tick() < tick)
        else {
            return Ok(());
        };
        let address = match head {
            DispatchedLsu::Load(issue) => issue.operands.effective_address,
            DispatchedLsu::Store(issue) => issue.operands.effective_address,
            DispatchedLsu::DirectStore(issue) => issue.operands.effective_address,
        };
        let roots = machine
            .spr_value(67)
            .zip(machine.spr_value(68))
            .ok_or(C220CoreError::LsuAddress { address })?;
        let mapped = C220ScalarMappedAddress::decode(address, roots.0, roots.1)
            .ok_or(C220CoreError::LsuAddress { address })?;
        let (instruction_id, request, second_request) = match head {
            DispatchedLsu::Load(issue) => {
                if mapped.memory != C220LsuMemory::External {
                    return Err(C220CoreError::UnsupportedTimedLsuAccess);
                }
                let Some((request, second)) = self.scheduler.admit_load(
                    tick,
                    issue.operands,
                    mapped,
                    self.config.partition_stack,
                )?
                else {
                    return Ok(());
                };
                self.commits
                    .admit(tick, C220LoadId(issue.instruction_id), request, second)?;
                (issue.instruction_id, request, second)
            }
            DispatchedLsu::DirectStore(issue) => {
                let Some(request) = self.scheduler.admit_direct_store(
                    tick,
                    issue.operands,
                    mapped,
                    self.config.partition_stack,
                    self.config.layout,
                )?
                else {
                    return Ok(());
                };
                self.pending.insert(request, issue);
                (issue.instruction_id, request, None)
            }
            DispatchedLsu::Store(issue) => {
                let Some(request) = self.scheduler.admit_store(
                    tick,
                    issue.operands,
                    mapped,
                    self.config.partition_stack,
                )?
                else {
                    return Ok(());
                };
                self.stores.insert(request, issue);
                (issue.instruction_id, request, None)
            }
        };
        self.ingress.pop_front();
        self.admissions.push(C220CoreLsuAdmission {
            instruction_id,
            request,
            second_request,
            tick,
            mapped,
        });
        Ok(())
    }
}
