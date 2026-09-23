use std::collections::VecDeque;

use super::{C220L0cError, C220L0cFragmentRequest, C220L0cScoreboard, C220L0cUnitFlagBlock};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0cReadRequest {
    pub id: u32,
    pub fragments: C220L0cFragmentRequest,
    pub data_type: u32,
    pub half_accumulator: bool,
}

impl C220L0cReadRequest {
    pub fn bank_mask(self, bank_count: u8) -> Result<u32, C220L0cError> {
        if bank_count == 0 || bank_count > 32 {
            return Err(C220L0cError::InvalidReadBankCount(bank_count));
        }
        let address = self.fragments.address;
        let mut mask = 0;
        for bank in 0..u32::from(bank_count) {
            let selected = if self.half_accumulator {
                ((address >> 9) ^ u64::from(bank)) & 1 == 0
            } else {
                let groups_of_four =
                    ((self.data_type >> 2) & 3) == 2 || matches!(self.data_type, 0 | 5);
                let pairs = matches!(self.data_type, 1..=3 | 7);
                (groups_of_four && ((u64::from(bank >> 2) ^ (address >> 9)) & 1 == 0))
                    || (pairs && ((u64::from(bank >> 1) ^ (address >> 8)) & 3 == 0))
            };
            mask |= u32::from(selected) << bank;
        }
        Ok(mask)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220L0cReadBlock {
    AwaitingBankAdmission,
    RequestOrder { head_id: u32 },
    UnitFlag(C220L0cUnitFlagBlock),
}

pub const C220_L0C_READ_TRANSPORT_CAPACITY: usize = 2;
pub const C220_L0C_READ_TRANSPORT_TICKS: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L0cReadTransit {
    pub request: C220L0cReadRequest,
    pub sent_tick: u64,
    pub ready_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220L0cReadPort {
    waiting: VecDeque<C220L0cReadTransit>,
    admitted: VecDeque<C220L0cReadRequest>,
    arbitration_tick: Option<u64>,
    observed_tick: Option<u64>,
}

impl C220L0cReadPort {
    pub fn request_ready(&self) -> bool {
        self.waiting.len() < C220_L0C_READ_TRANSPORT_CAPACITY
    }

    pub fn incoming(&self) -> &VecDeque<C220L0cReadTransit> {
        &self.waiting
    }

    /// A blocked send leaves the request with the producer.
    pub fn send_request(
        &mut self,
        tick: u64,
        request: C220L0cReadRequest,
    ) -> Result<bool, C220L0cError> {
        self.check_time(tick)?;
        if !self.request_ready() {
            self.observed_tick = Some(tick);
            return Ok(false);
        }
        let ready_tick = tick
            .checked_add(C220_L0C_READ_TRANSPORT_TICKS)
            .ok_or(C220L0cError::TimeOverflow)?;
        self.waiting.push_back(C220L0cReadTransit {
            request,
            sent_tick: tick,
            ready_tick,
        });
        self.observed_tick = Some(tick);
        Ok(true)
    }

    fn check_time(&self, tick: u64) -> Result<(), C220L0cError> {
        if let Some(previous) = self.observed_tick
            && tick < previous
        {
            return Err(C220L0cError::ReadTimeReversed {
                requested: tick,
                previous,
            });
        }
        Ok(())
    }

    pub fn queue_lengths(&self) -> (usize, usize) {
        (self.waiting.len(), self.admitted.len())
    }

    pub fn arbitrate(
        &mut self,
        banks: &C220L0cReadBanks,
        bank_count: u8,
    ) -> Result<Option<u32>, C220L0cError> {
        self.check_time(banks.tick())?;
        if self.arbitration_tick == Some(banks.tick()) {
            return Ok(None);
        }
        let conflict = self
            .waiting
            .front()
            .filter(|transit| transit.ready_tick <= banks.tick())
            .map(|transit| {
                transit
                    .request
                    .bank_mask(bank_count)
                    .map(|mask| banks.conflicts(mask))
            })
            .transpose()?;
        self.arbitration_tick = Some(banks.tick());
        self.observed_tick = Some(banks.tick());
        if conflict == Some(0) {
            self.admitted.push_back(
                self.waiting
                    .pop_front()
                    .expect("waiting read exists")
                    .request,
            );
        }
        Ok(conflict)
    }

    pub fn complete_read(
        &mut self,
        id: u32,
        tick: u64,
        data_latency: u32,
        scoreboard: &mut C220L0cScoreboard,
    ) -> Result<Result<u64, C220L0cReadBlock>, C220L0cError> {
        self.check_time(tick)?;
        self.observed_tick = Some(tick);
        let Some(request) = self.admitted.front() else {
            return Ok(Err(C220L0cReadBlock::AwaitingBankAdmission));
        };
        if request.id != id {
            return Ok(Err(C220L0cReadBlock::RequestOrder {
                head_id: request.id,
            }));
        }
        let ready = tick
            .checked_add(u64::from(data_latency))
            .ok_or(C220L0cError::TimeOverflow)?;
        scoreboard.advance_to(tick);
        if let Err(block) = scoreboard.admit(request.fragments, tick)? {
            return Ok(Err(C220L0cReadBlock::UnitFlag(block)));
        }
        self.admitted.pop_front();
        Ok(Ok(ready))
    }
}

const CUBE_BANK_SEQUENCE: [u16; 7] = [0x0001, 0x0012, 0x0124, 0x1248, 0x2480, 0x4800, 0x8000];

/// Read-bank occupancy indexed by the cycle at which a request reaches L0C.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct C220L0cReadBanks {
    tick: u64,
    masks: [u32; 8],
}

impl C220L0cReadBanks {
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    pub const fn occupied_mask(&self) -> u32 {
        self.masks[0]
    }

    pub const fn future_masks(&self) -> &[u32; 8] {
        &self.masks
    }

    pub fn advance_to(&mut self, tick: u64) -> Result<(), C220L0cError> {
        let elapsed = tick
            .checked_sub(self.tick)
            .ok_or(C220L0cError::ReadTimeReversed {
                requested: tick,
                previous: self.tick,
            })?;
        let elapsed = elapsed.min(self.masks.len() as u64) as usize;
        self.masks.rotate_left(elapsed);
        let remaining = self.masks.len() - elapsed;
        self.masks[remaining..].fill(0);
        self.tick = tick;
        Ok(())
    }

    pub fn accept_cube_read(
        &mut self,
        arrival_tick: u64,
        address: u64,
        half_accumulator: bool,
    ) -> Result<(), C220L0cError> {
        self.advance_to(arrival_tick)?;
        self.reserve(address, half_accumulator, 0);
        Ok(())
    }

    pub fn send_cube_read(
        &mut self,
        issue_tick: u64,
        address: u64,
        half_accumulator: bool,
    ) -> Result<(), C220L0cError> {
        issue_tick
            .checked_add(1)
            .ok_or(C220L0cError::TimeOverflow)?;
        self.advance_to(issue_tick)?;
        self.reserve(address, half_accumulator, 1);
        Ok(())
    }

    fn reserve(&mut self, address: u64, half_accumulator: bool, offset: usize) {
        for (occupied, pattern) in self.masks[offset..].iter_mut().zip(CUBE_BANK_SEQUENCE) {
            let pattern = u32::from(pattern);
            *occupied |= if half_accumulator {
                pattern << (16 * ((address >> 8) & 1))
            } else {
                pattern | (pattern << 16)
            };
        }
    }

    pub fn conflicts(&self, requested_banks: u32) -> u32 {
        self.occupied_mask() & requested_banks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incoming_transport_holds_capacity_until_bank_admission() {
        use crate::sim::c220::cube::C220CubeL0cAccess;

        let request = C220L0cReadRequest {
            id: 1,
            fragments: C220L0cFragmentRequest {
                address: 0,
                bytes: 512,
                access: C220CubeL0cAccess::Read,
                check_unit_flags: false,
                update_unit_flags: false,
            },
            data_type: 0,
            half_accumulator: false,
        };
        let mut port = C220L0cReadPort::default();
        let mut banks = C220L0cReadBanks::default();
        assert!(port.send_request(0, request).unwrap());
        assert!(
            port.send_request(0, C220L0cReadRequest { id: 2, ..request })
                .unwrap()
        );
        assert!(
            !port
                .send_request(0, C220L0cReadRequest { id: 3, ..request })
                .unwrap()
        );
        assert_eq!(port.arbitrate(&banks, 32).unwrap(), None);
        assert_eq!(port.queue_lengths(), (2, 0));
        assert_eq!(port.incoming()[0].ready_tick, 1);

        banks.accept_cube_read(1, 0, false).unwrap();
        assert_ne!(port.arbitrate(&banks, 32).unwrap(), Some(0));
        assert!(!port.request_ready());
        banks.advance_to(8).unwrap();
        assert_eq!(port.arbitrate(&banks, 32).unwrap(), Some(0));
        assert_eq!(port.queue_lengths(), (1, 1));
        assert!(
            port.send_request(8, C220L0cReadRequest { id: 3, ..request })
                .unwrap()
        );
        assert_eq!(port.arbitrate(&banks, 32).unwrap(), None);
        banks.advance_to(9).unwrap();
        assert_eq!(port.arbitrate(&banks, 32).unwrap(), Some(0));
        assert_eq!(port.queue_lengths(), (1, 2));
        assert_eq!(port.incoming()[0].request.id, 3);
        let saved = port.clone();
        assert!(matches!(
            port.send_request(8, request),
            Err(C220L0cError::ReadTimeReversed { .. })
        ));
        assert!(matches!(
            port.send_request(u64::MAX, request),
            Err(C220L0cError::TimeOverflow)
        ));
        assert_eq!(port, saved);
    }

    #[test]
    fn read_port_waits_for_banks_and_preserves_request_order() {
        use crate::sim::c220::cube::C220CubeL0cAccess;
        use crate::sim::c220::memory::C220L0c;
        let request = C220L0cReadRequest {
            id: 1,
            fragments: C220L0cFragmentRequest {
                address: 0,
                bytes: 512,
                access: C220CubeL0cAccess::Read,
                check_unit_flags: false,
                update_unit_flags: false,
            },
            data_type: 0,
            half_accumulator: false,
        };
        assert_eq!(request.bank_mask(32).unwrap(), 0x0f0f_0f0f);
        let mut memory = C220L0c::new(131072, 12).unwrap();
        memory
            .read_banks_mut()
            .accept_cube_read(0, 0, false)
            .unwrap();
        assert!(memory.read_port_mut().send_request(0, request).unwrap());
        assert_eq!(
            memory.poll_read(0, 1, 32, 5).unwrap(),
            Err(C220L0cReadBlock::AwaitingBankAdmission)
        );
        assert_eq!(
            memory.poll_read(7, 2, 32, 5).unwrap(),
            Err(C220L0cReadBlock::RequestOrder { head_id: 1 })
        );
        assert_eq!(memory.poll_read(7, 1, 32, 5).unwrap(), Ok(12));
        assert!(
            memory
                .read_port_mut()
                .send_request(7, C220L0cReadRequest { id: 2, ..request })
                .unwrap()
        );
        assert_eq!(
            memory.poll_read(7, 2, 32, 5).unwrap(),
            Err(C220L0cReadBlock::AwaitingBankAdmission)
        );
        assert_eq!(memory.poll_read(8, 2, 32, 5).unwrap(), Ok(13));
    }

    #[test]
    fn overlapping_reads_preserve_age_and_accumulator_half() {
        let mut banks = C220L0cReadBanks::default();
        banks.accept_cube_read(10, 0, false).unwrap();
        assert_eq!(banks.occupied_mask(), 0x0001_0001);
        banks.accept_cube_read(11, 256, true).unwrap();
        assert_eq!(banks.occupied_mask(), 0x0013_0012);
        assert_eq!(banks.conflicts(0xffff), 0x12);
        banks.advance_to(16).unwrap();
        assert_eq!(banks.occupied_mask(), 0xc800_8000);
        banks.advance_to(17).unwrap();
        assert_eq!(banks.occupied_mask(), 0x8000_0000);
        banks.advance_to(18).unwrap();
        assert_eq!(banks.future_masks(), &[0; 8]);
        assert!(banks.advance_to(17).is_err());
        assert_eq!(banks.tick(), 18);
    }
}
