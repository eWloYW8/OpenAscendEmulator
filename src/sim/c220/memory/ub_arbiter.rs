use thiserror::Error;

use super::C220UbBlockProgress;

const BANK_BYTES: u64 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbBank {
    pub group: u8,
    pub id: u8,
}

impl C220UbBank {
    pub const fn from_address(address: u64) -> Self {
        let group = ((address >> 5) & 0xf) as u8;
        let id = group | ((address >> 12) & 0x30) as u8;
        Self { group, id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220UbRequestError {
    #[error("UB access beginning at {address:#x} with {bytes} bytes overflows")]
    AddressOverflow { address: u64, bytes: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbBlock {
    pub address: u64,
    pub bytes: u8,
    pub bank: C220UbBank,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220UbRequest {
    blocks: Vec<C220UbBlock>,
    grants: Vec<Option<u64>>,
    progress: Vec<C220UbBlockProgress>,
    partial_writes: Vec<bool>,
}

impl C220UbRequest {
    pub fn from_accesses(accesses: &[(u64, usize)]) -> Result<Self, C220UbRequestError> {
        Self::build(
            accesses
                .iter()
                .map(|&(address, bytes)| (address, bytes, false, false)),
        )
    }

    /// Each access carries whether its entire element mask is enabled. A
    /// masked access is still split across its full physical address span.
    pub fn from_writes(accesses: &[(u64, usize, bool)]) -> Result<Self, C220UbRequestError> {
        Self::build(
            accesses
                .iter()
                .map(|&(address, bytes, full_mask)| (address, bytes, true, full_mask)),
        )
    }

    fn build(
        accesses: impl IntoIterator<Item = (u64, usize, bool, bool)>,
    ) -> Result<Self, C220UbRequestError> {
        let mut blocks = Vec::new();
        let mut partial_writes = Vec::new();
        for (address, bytes, write, full_mask) in accesses {
            let end = address
                .checked_add(bytes as u64)
                .ok_or(C220UbRequestError::AddressOverflow { address, bytes })?;
            let mut at = address;
            while at < end {
                let boundary = at.saturating_add(BANK_BYTES - at % BANK_BYTES);
                let next = boundary.min(end);
                blocks.push(C220UbBlock {
                    address: at,
                    bytes: (next - at) as u8,
                    bank: C220UbBank::from_address(at),
                });
                partial_writes
                    .push(write && (!full_mask || at % BANK_BYTES != 0 || next - at != BANK_BYTES));
                at = next;
            }
        }
        Ok(Self {
            grants: vec![None; blocks.len()],
            progress: vec![C220UbBlockProgress::Waiting; blocks.len()],
            partial_writes,
            blocks,
        })
    }

    pub fn blocks(&self) -> &[C220UbBlock] {
        &self.blocks
    }

    pub fn grants(&self) -> &[Option<u64>] {
        &self.grants
    }

    pub fn progress(&self) -> &[C220UbBlockProgress] {
        &self.progress
    }

    pub fn is_complete(&self) -> bool {
        self.grants.iter().all(Option::is_some)
    }

    pub fn completion_tick(&self) -> Option<u64> {
        self.grants.iter().copied().flatten().max()
    }

    /// Projects this request in isolation, preserving completed grants and
    /// partial-write delay state. Other UB masters can delay the result.
    pub fn projected_write_completion(&self, start_tick: u64) -> Option<u64> {
        let mut request = self.clone();
        let mut tick = start_tick;
        while !request.is_complete() {
            C220UbCycle::arbitrate(tick, Some(&mut request), None, None);
            if request.is_complete() {
                break;
            }
            tick = tick.checked_add(1)?;
        }
        request.completion_tick()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220UbPort {
    VectorWrite,
    VectorRead0,
    VectorRead1,
    VectorReadDestination,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbDecision {
    pub port: C220UbPort,
    pub block_index: usize,
    pub block: C220UbBlock,
    pub granted: bool,
    pub second_grant: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220UbCycle {
    pub tick: u64,
    pub bank_mask: u64,
    pub read_group_mask: u16,
    pub write_group_mask: u16,
    pub decisions: Vec<C220UbDecision>,
}

impl C220UbCycle {
    pub fn arbitrate(
        tick: u64,
        write: Option<&mut C220UbRequest>,
        read0: Option<&mut C220UbRequest>,
        read1: Option<&mut C220UbRequest>,
    ) -> Self {
        Self::arbitrate_with_destination(tick, write, read0, read1, None)
    }

    pub fn arbitrate_with_destination(
        tick: u64,
        write: Option<&mut C220UbRequest>,
        read0: Option<&mut C220UbRequest>,
        read1: Option<&mut C220UbRequest>,
        read_destination: Option<&mut C220UbRequest>,
    ) -> Self {
        let mut cycle = Self {
            tick,
            bank_mask: 0,
            read_group_mask: 0,
            write_group_mask: 0,
            decisions: Vec::new(),
        };
        if let Some(request) = write {
            cycle.arbitrate_port(C220UbPort::VectorWrite, request);
        }
        if let Some(request) = read0 {
            cycle.arbitrate_port(C220UbPort::VectorRead0, request);
        }
        if let Some(request) = read1 {
            cycle.arbitrate_port(C220UbPort::VectorRead1, request);
        }
        if let Some(request) = read_destination {
            cycle.arbitrate_port(C220UbPort::VectorReadDestination, request);
        }
        cycle
    }

    fn arbitrate_port(&mut self, port: C220UbPort, request: &mut C220UbRequest) {
        let group_mask = match port {
            C220UbPort::VectorWrite => &mut self.write_group_mask,
            C220UbPort::VectorRead0
            | C220UbPort::VectorRead1
            | C220UbPort::VectorReadDestination => &mut self.read_group_mask,
        };
        for (index, &block) in request.blocks.iter().enumerate() {
            let Some(second_grant) = request.progress[index].pending_grant() else {
                continue;
            };
            let bank_bit = 1_u64 << block.bank.id;
            let group_bit = 1_u16 << block.bank.group;
            let granted = self.bank_mask & bank_bit == 0 && *group_mask & group_bit == 0;
            if granted {
                request.progress[index].grant(self.tick, request.partial_writes[index]);
                request.grants[index] = request.progress[index].completion_tick();
                self.bank_mask |= bank_bit;
                *group_mask |= group_bit;
            }
            self.decisions.push(C220UbDecision {
                port,
                block_index: index,
                block,
                granted,
                second_grant,
            });
        }
        for progress in &mut request.progress {
            progress.finish_tick(self.tick);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_id_uses_group_and_upper_region_bits() {
        assert_eq!(C220UbBank::from_address(0), C220UbBank { group: 0, id: 0 });
        assert_eq!(
            C220UbBank::from_address(0x20),
            C220UbBank { group: 1, id: 1 }
        );
        assert_eq!(
            C220UbBank::from_address(0x10020),
            C220UbBank { group: 1, id: 17 }
        );
    }

    #[test]
    fn splits_at_bank_boundaries_and_rejects_address_overflow() {
        let request = C220UbRequest::from_accesses(&[(31, 3), (0x10020, 32)]).unwrap();
        assert_eq!(
            request.blocks(),
            [
                C220UbBlock {
                    address: 31,
                    bytes: 1,
                    bank: C220UbBank::from_address(31),
                },
                C220UbBlock {
                    address: 32,
                    bytes: 2,
                    bank: C220UbBank::from_address(32),
                },
                C220UbBlock {
                    address: 0x10020,
                    bytes: 32,
                    bank: C220UbBank::from_address(0x10020),
                }
            ]
        );
        assert_eq!(
            C220UbRequest::from_accesses(&[(u64::MAX, 2)]),
            Err(C220UbRequestError::AddressOverflow {
                address: u64::MAX,
                bytes: 2,
            })
        );
    }

    #[test]
    fn unaligned_full_mask_writes_require_two_grants_per_split() {
        let mut write = C220UbRequest::from_writes(&[(31, 32, true)]).unwrap();
        assert_eq!(write.projected_write_completion(0), Some(7));
        let first = C220UbCycle::arbitrate(0, Some(&mut write), None, None);
        assert_eq!(first.bank_mask, 3);
        assert_eq!(write.grants(), [None, None]);
        for tick in 1..=6 {
            assert!(
                C220UbCycle::arbitrate(tick, Some(&mut write), None, None)
                    .decisions
                    .is_empty()
            );
        }
        let second = C220UbCycle::arbitrate(7, Some(&mut write), None, None);
        assert!(
            second
                .decisions
                .iter()
                .all(|decision| decision.granted && decision.second_grant)
        );
        assert_eq!(write.grants(), [Some(7), Some(7)]);
        assert_eq!(write.projected_write_completion(8), Some(7));
    }

    #[test]
    fn writes_precede_reads_and_read_ports_share_group_mask() {
        let mut write = C220UbRequest::from_accesses(&[(0x20, 32)]).unwrap();
        let mut read0 = C220UbRequest::from_accesses(&[(0x20, 32), (0x40, 32)]).unwrap();
        let mut read1 = C220UbRequest::from_accesses(&[(0x10020, 32), (0x60, 32)]).unwrap();
        let first =
            C220UbCycle::arbitrate(10, Some(&mut write), Some(&mut read0), Some(&mut read1));
        assert_eq!(first.bank_mask.count_ones(), 4);
        assert_eq!(first.write_group_mask.count_ones(), 1);
        assert_eq!(first.read_group_mask.count_ones(), 3);
        assert_eq!(write.grants(), [Some(10)]);
        assert_eq!(read0.grants(), [None, Some(10)]);
        assert_eq!(read1.grants(), [Some(10), Some(10)]);

        let second = C220UbCycle::arbitrate(11, None, Some(&mut read0), None);
        assert_eq!(second.decisions.len(), 1);
        assert_eq!(read0.grants(), [Some(11), Some(10)]);
        assert!(read0.is_complete());
        assert_eq!(read0.completion_tick(), Some(11));
    }
}
