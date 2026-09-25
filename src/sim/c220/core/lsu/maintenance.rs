use super::*;
use crate::isa::flow::{DcciInstruction, DcciStep};
use crate::sim::c220::scalar::lsu::cache::C220CacheMaintenanceTarget;
use crate::sim::c220::scalar::lsu::scheduler::{
    C220LsuMaintenanceCompletion, C220LsuMaintenanceScope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CoreMaintenanceIssue {
    pub instruction_id: u64,
    pub tick: u64,
    pub step: DcciStep,
    pub scope: C220LsuMaintenanceScope,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CoreMaintenanceCompletion {
    pub issue: C220CoreMaintenanceIssue,
    pub data: C220LsuMaintenanceCompletion,
    pub retire_tick: u64,
}

impl C220Core {
    pub fn pending_maintenance_instructions(
        &self,
    ) -> impl Iterator<Item = &C220CoreMaintenanceIssue> {
        self.lsu.iter().flat_map(|lsu| {
            lsu.maintenance
                .values()
                .map(|entry| &entry.0)
                .chain(lsu.ingress.iter().filter_map(|entry| match entry {
                    DispatchedLsu::Maintenance(issue) => Some(issue),
                    _ => None,
                }))
        })
    }

    pub fn take_maintenance_completions(&mut self) -> Vec<C220CoreMaintenanceCompletion> {
        self.lsu
            .as_mut()
            .map(|lsu| std::mem::take(&mut lsu.maintenance_completions))
            .unwrap_or_default()
    }

    pub(in crate::sim::c220::core) fn step_maintenance_at(
        &mut self,
        tick: u64,
        instruction: DcciInstruction,
    ) -> Result<C220CoreStep, C220CoreError> {
        let pc = self.state.scalar().pc();
        if self.state.scalar().is_halted() {
            return Err(crate::sim::c220::state::C220ExecutionError::ProgramEnded { pc }.into());
        }
        let machine = self.state.scalar().machine();
        let step = instruction.resolve(pc, machine.xregs());
        let scope = if step.entire_cache {
            C220LsuMaintenanceScope::All {
                target: match step.operation_field {
                    0 => C220CacheMaintenanceTarget::All,
                    1 => C220CacheMaintenanceTarget::Ub,
                    2 => C220CacheMaintenanceTarget::External,
                    _ => C220CacheMaintenanceTarget::Atomic,
                },
            }
        } else {
            C220LsuMaintenanceScope::Line {
                address: step.effective_address & 0x0000_ffff_ffff_ffff,
                partition_address: step.effective_address,
            }
        };
        let next = tick.checked_add(1).ok_or(C220CoreError::TimeOverflow)?;
        let lsu = self.lsu.as_mut().ok_or(C220CoreError::LsuUnconfigured)?;
        if lsu.cache.is_none() {
            return Err(C220CoreError::LsuUnconfigured);
        }
        if lsu.ingress.len() == 2 {
            return Ok(C220CoreStep::Stalled(C220Stall {
                tick,
                pc,
                resume_tick: next,
                cause: C220StallCause::LsuDependency,
            }));
        }
        let issue = C220CoreMaintenanceIssue {
            instruction_id: self.next_instruction_id,
            tick,
            step,
            scope,
        };
        lsu.ingress.push_back(DispatchedLsu::Maintenance(issue));
        lsu.next_tick = Some(lsu.next_tick.map_or(next, |prior| prior.min(next)));
        self.state.commit_c220_sequential_issue();
        Ok(C220CoreStep::Executed {
            tick,
            instruction: C220CoreInstruction::Maintenance(issue),
        })
    }
}
