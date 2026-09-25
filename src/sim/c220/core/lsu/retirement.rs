use super::*;
use crate::sim::c220::scalar::lsu::commit::C220LsuRetirement;
use crate::sim::common::scalar::ScalarMachine;

impl C220Core {
    pub fn lsu_retirement_occupancy(&self) -> usize {
        self.lsu
            .as_ref()
            .map_or(0, |lsu| lsu.commits.retirement_occupancy())
    }
}

impl CoreLsu {
    pub(super) fn retire_at(
        &mut self,
        tick: u64,
        machine: &mut ScalarMachine,
        memory: &mut crate::memory::mapped::MappedMemory,
    ) -> Result<(), C220CoreError> {
        let atomic_result = self
            .commits
            .ready_atomic_at(tick)?
            .map(|request| {
                let mut operands = self.atomics[&request].0.operands;
                let spr = |spr| {
                    machine.spr_value(spr).ok_or(
                        crate::sim::common::scalar::ScalarInstructionError::from(
                            crate::sim::common::scalar::ScalarMachineError::SprValueUnavailable {
                                pc: operands.pc,
                                spr,
                            },
                        ),
                    )
                };
                operands.control = spr(3)?;
                operands.atomic_control = spr(90)?;
                operands.local_root = spr(67)?;
                let result = operands.execute(memory, self.config.atomic_fp16_rounding)?;
                Ok::<_, C220CoreError>(result)
            })
            .transpose()?;
        match self.commits.retire_next_at(tick, machine)? {
            Some(C220LsuRetirement::AtomicStore {
                request,
                retire_tick,
            }) => {
                let (issue, data) = self.atomics.remove(&request).expect("issued atomic store");
                let result = atomic_result.expect("ready atomic result");
                if let Some(base) = issue.operands.updated_base {
                    machine
                        .set_xreg(issue.operands.instruction().base_register, base)
                        .map_err(crate::sim::common::scalar::ScalarInstructionError::from)?;
                }
                self.atomic_completions.push(C220CoreAtomicCompletion {
                    issue,
                    result,
                    data: data.expect("completed atomic store"),
                    retire_tick,
                });
            }
            Some(C220LsuRetirement::Maintenance {
                request,
                retire_tick,
            }) => {
                let (issue, data) = self
                    .maintenance
                    .remove(&request)
                    .expect("issued maintenance");
                self.maintenance_completions
                    .push(C220CoreMaintenanceCompletion {
                        issue,
                        data: data.expect("completed maintenance"),
                        retire_tick,
                    });
            }
            Some(C220LsuRetirement::RepeatedLoad(notification)) => {
                self.cache
                    .as_mut()
                    .expect("issued load cache")
                    .repeated_notifications
                    .push(notification);
            }
            Some(C220LsuRetirement::Load(retirement)) => {
                let cache = self.cache.as_mut().expect("issued load cache");
                let issue = cache
                    .pending
                    .remove(&retirement.instruction)
                    .expect("issued load");
                cache
                    .completions
                    .push(C220CoreLoadCompletion { issue, retirement });
            }
            Some(C220LsuRetirement::Store {
                data,
                retire_tick,
                first_request,
                final_notification,
                repeated_notification,
                responses,
            }) => {
                let issue = if final_notification {
                    self.stores.remove(&first_request).expect("issued store")
                } else {
                    self.stores[&first_request]
                };
                self.store_completions.push(C220CoreStoreCompletion {
                    issue,
                    data,
                    retire_tick,
                    repeated_notification,
                    responses,
                });
            }
            Some(C220LsuRetirement::DirectStore {
                request,
                write,
                response_tick,
                retire_tick,
            }) => {
                let issue = self.pending.remove(&request).expect("issued direct store");
                self.completions.push(C220CoreLsuCompletion {
                    issue,
                    request,
                    write,
                    response_tick,
                    tick: retire_tick,
                });
            }
            None => {}
        }
        Ok(())
    }
}
