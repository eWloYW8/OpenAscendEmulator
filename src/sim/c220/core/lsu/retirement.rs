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
    ) -> Result<(), C220CoreError> {
        match self.commits.retire_next_at(tick, machine)? {
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
