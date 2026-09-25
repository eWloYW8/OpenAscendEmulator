use super::super::cache::{C220CacheLocation, C220DataCache};
use super::super::commit::{C220LsuCommitError, C220LsuCommitLane};
use super::super::store_buffer::{C220LsuCompletion, C220LsuPairPart};
use super::*;
use crate::sim::c220::scalar::{C220LoadOperands, C220ScalarMappedAddress};

mod data;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220LsuLoadPath {
    Cache,
    StoreForward,
    CacheAndStore,
    Refill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220LsuLoadValue {
    pub request: C220LsuRequestId,
    pub tick: u64,
    pub operands: C220LoadOperands,
    pub mapped: C220ScalarMappedAddress,
    pub value: u64,
    pub second_value: Option<u64>,
    pub path: C220LsuLoadPath,
    pub part: C220LsuPairPart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PendingLoad {
    operands: C220LoadOperands,
    mapped: C220ScalarMappedAddress,
    line: C220LsuLineKey,
    partition_address: u64,
    offset: usize,
    lookup: Option<Option<C220CacheLocation>>,
    part: C220LsuPairPart,
}

impl C220LsuRequestScheduler {
    pub fn admit_load(
        &mut self,
        tick: u64,
        operands: C220LoadOperands,
        mapped: C220ScalarMappedAddress,
        partition_stack: bool,
    ) -> Result<Option<(C220LsuRequestId, Option<C220LsuRequestId>)>, C220LsuSchedulerError> {
        let size = self.misses.line_bytes() as u64;
        let physical_address = mapped.address & 0x0000_ffff_ffff_ffff;
        let offset = (physical_address % size) as usize;
        let width = usize::from(operands.width_bytes);
        let split = operands.second_destination.is_some()
            && physical_address >> 6 != physical_address.wrapping_add(width as u64) >> 6;
        let second_address = physical_address.wrapping_add(width as u64);
        let bytes = if split {
            width
        } else {
            operands.access_bytes()
        };
        if ![1, 2, 4, 8].contains(&operands.width_bytes)
            || offset + bytes > size as usize
            || (split && second_address % size + width as u64 > size)
        {
            return Err(C220LsuStoreError::InvalidRange.into());
        }
        let line = C220LsuLineKey {
            address: physical_address / size * size,
            memory: mapped.memory,
        };
        let second_line = C220LsuLineKey {
            address: second_address / size * size,
            memory: mapped.memory,
        };
        let Some((id, second)) = self.admit(
            tick,
            C220LsuRequest {
                line,
                access: C220LsuAccess::Load,
            },
            split.then_some(C220LsuRequest {
                line: second_line,
                access: C220LsuAccess::Load,
            }),
        )?
        else {
            return Ok(None);
        };
        self.pending_loads.insert(
            id,
            PendingLoad {
                operands,
                mapped,
                line,
                offset,
                partition_address: mapped.cache_address(partition_stack),
                lookup: None,
                part: if split {
                    C220LsuPairPart::First
                } else {
                    C220LsuPairPart::Both
                },
            },
        );
        if let Some(second) = second {
            let second_mapped = C220ScalarMappedAddress {
                address: second_address,
                ..mapped
            };
            self.pending_loads.insert(
                second,
                PendingLoad {
                    operands,
                    mapped: second_mapped,
                    line: second_line,
                    offset: (second_address % size) as usize,
                    partition_address: second_mapped.cache_address(partition_stack),
                    lookup: None,
                    part: C220LsuPairPart::Second,
                },
            );
        }
        Ok(Some((id, second)))
    }

    pub fn pending_load(&self, id: C220LsuRequestId) -> Option<&C220LoadOperands> {
        self.pending_loads.get(&id).map(|pending| &pending.operands)
    }

    /// Data availability, not architectural retirement. The core owns final
    /// register updates and release of dependent instructions.
    pub fn take_load_values(&mut self) -> Vec<C220LsuLoadValue> {
        let mut loads = Vec::new();
        self.values.retain(|value| match value {
            C220LsuValue::Load(data) => {
                loads.push(*data);
                false
            }
            C220LsuValue::Store(_) => true,
        });
        loads
    }

    /// Deliver completed data in order, retaining the rejected completion and
    /// its successors if register commit or retirement admission fails.
    pub fn deliver_values(
        &mut self,
        tick: u64,
        commits: &mut C220LsuCommitLane,
        machine: &mut crate::sim::common::scalar::ScalarMachine,
    ) -> Result<usize, C220LsuCommitError> {
        let mut accepted = 0;
        while let Some(value) = self.values.front().copied() {
            match value {
                C220LsuValue::Load(data) => commits.complete_data_at(tick, data, machine)?,
                C220LsuValue::Store(data) => commits.complete_store_at(tick, data)?,
            }
            self.values.pop_front();
            accepted += 1;
        }
        Ok(accepted)
    }

    pub fn advance_with_cache(
        &mut self,
        stage: C220LsuStage,
        tick: u64,
        external: C220LsuExternalHazards,
        cache: &mut C220DataCache,
    ) -> Result<C220LsuStageOutcome, C220LsuSchedulerError> {
        if cache.line_bytes() != self.misses.line_bytes() {
            return Err(C220LsuSchedulerError::LineGeometry);
        }
        self.prepare_maintenance_stage(stage, tick, cache)?;
        let pending = self
            .pipeline
            .head(stage)
            .filter(|head| head.ready_tick <= tick)
            .and_then(|head| {
                self.pending_loads
                    .get(&head.request)
                    .copied()
                    .map(|load| (head.request, load))
            });
        let mut result = None;
        if stage == C220LsuStage::M2 {
            self.validate_store_m2(tick)?;
        }
        if let Some((id, pending)) = pending
            && stage == C220LsuStage::M2
            && self.hazard(self.requests[&id], external).is_none()
        {
            let location = pending
                .lookup
                .ok_or(C220LsuSchedulerError::MissingLoadLookup)?;
            let width = pending.access_bytes();
            let store = self.stores.entry(pending.line);
            let covered = store
                .map(|entry| entry.covered_bytes(pending.offset, width))
                .transpose()?
                .unwrap_or(0);
            if location.is_some() || covered == width {
                let (values, forwarded) = pending.read_data(
                    location.map(|location| cache.line(location)).transpose()?,
                    store,
                    covered,
                )?;
                let path = match (location.is_some(), forwarded) {
                    (true, false) => C220LsuLoadPath::Cache,
                    (true, true) => C220LsuLoadPath::CacheAndStore,
                    _ => C220LsuLoadPath::StoreForward,
                };
                result = Some((values, path));
            } else {
                self.reads.check_enqueue()?;
                tick.checked_add(1).ok_or(C220LsuReadError::Overflow)?;
            }
        }
        let outcome = self.advance_impl(stage, tick, external)?;
        if let C220LsuStageProgress::Advanced(id) = outcome.progress {
            self.advance_store_data(stage, tick, id, cache)?;
            self.advance_maintenance_stage(stage, tick, id, cache)?;
        }
        if let C220LsuStageProgress::Advanced(id) = outcome.progress
            && let Some((pending_id, pending)) = pending
        {
            debug_assert_eq!(pending_id, id);
            match stage {
                C220LsuStage::M0 => {
                    let location = cache.lookup_partitioned(
                        pending.line.address,
                        pending.partition_address,
                        pending.line.memory,
                    );
                    self.pending_loads
                        .get_mut(&id)
                        .expect("admitted load")
                        .lookup = Some(location.or(pending.lookup.flatten()));
                }
                C220LsuStage::M2 => {
                    if let Some((value, path)) = result {
                        self.finish_load(id, pending, tick, value, path);
                    } else {
                        self.enqueue_load_miss(pending.line, id, pending.partition_address)?;
                    }
                }
                C220LsuStage::M1 => {}
            }
        }
        Ok(outcome)
    }

    fn finish_load(
        &mut self,
        id: C220LsuRequestId,
        pending: PendingLoad,
        tick: u64,
        values: [u64; 2],
        path: C220LsuLoadPath,
    ) {
        self.pending_loads.remove(&id);
        self.values.push_back(C220LsuValue::Load(C220LsuLoadValue {
            request: id,
            tick,
            operands: pending.operands,
            mapped: pending.mapped,
            value: values[0],
            second_value: pending.operands.second_destination.map(|_| values[1]),
            path,
            part: pending.part,
        }));
    }

    pub(super) fn resolve_load_values(
        &mut self,
        tick: u64,
        key: C220LsuLineKey,
        notifications: &[C220LsuCompletion],
        line: &[u8],
    ) {
        for notification in notifications {
            if let C220LsuCompletion::Load(id) = *notification
                && let Some(pending) = self.pending_loads.get(&id).copied()
            {
                debug_assert_eq!(pending.line, key);
                self.finish_load(
                    id,
                    pending,
                    tick,
                    pending.read_line(line),
                    C220LsuLoadPath::Refill,
                );
            }
        }
    }
}
