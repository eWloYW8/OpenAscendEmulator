use super::{C220L1Error, C220L1Port};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Geometry {
    bank_width: u32,
    bank_count: u32,
    group_count: u32,
    group_offset: u32,
}

impl C220L1Geometry {
    pub fn new(
        bank_width: u32,
        bank_count: u32,
        group_count: u32,
        group_offset: u32,
    ) -> Result<Self, C220L1Error> {
        if bank_width == 0
            || !(1..=32).contains(&bank_count)
            || group_count == 0
            || group_count > 64 / bank_count
            || group_offset >= 64
        {
            return Err(C220L1Error::UnsupportedGeometry);
        }
        Ok(Self {
            bank_width,
            bank_count,
            group_count,
            group_offset,
        })
    }

    pub const fn bank_width(self) -> u32 {
        self.bank_width
    }

    pub const fn bank_count(self) -> u32 {
        self.bank_count
    }

    pub const fn group_count(self) -> u32 {
        self.group_count
    }

    pub const fn group_offset(self) -> u32 {
        self.group_offset
    }

    fn base_mask(self, access: C220L1Access) -> u64 {
        let start = (access.address as u32 / self.bank_width) % self.bank_count;
        let count = access.bytes.wrapping_sub(1).wrapping_add(self.bank_width) / self.bank_width;
        let end = start.wrapping_add(count);
        let mut mask = 0;
        if end > start {
            for offset in 0..(end - start).min(self.bank_count) {
                mask |= 1_u64 << ((start + offset) % self.bank_count);
            }
        }
        mask
    }

    fn group_shift(self, address: u64) -> u32 {
        if self.group_offset == 0 {
            0
        } else {
            ((address >> self.group_offset) as u32 & (self.group_count - 1)) * self.bank_count
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Access {
    pub address: u64,
    pub bytes: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220L1Decision {
    pub access: C220L1Access,
    pub bank_mask: u64,
    pub granted: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220L1Arbiter {
    geometry: C220L1Geometry,
    cache_tags: Vec<Option<u32>>,
}

impl C220L1Arbiter {
    pub fn new(geometry: C220L1Geometry) -> Self {
        Self {
            geometry,
            cache_tags: vec![None; geometry.bank_count as usize],
        }
    }

    pub const fn geometry(&self) -> C220L1Geometry {
        self.geometry
    }

    pub fn cache_tags(&self) -> &[Option<u32>] {
        &self.cache_tags
    }

    pub fn write_bank_mask(&self, access: C220L1Access) -> u64 {
        self.geometry.base_mask(access) << self.geometry.group_shift(access.address)
    }

    pub fn read_bank_mask(&self, access: C220L1Access) -> u64 {
        let mut mask = self.geometry.base_mask(access);
        let mut tag = (access.address / u64::from(self.geometry.bank_width)) as u32;
        for (bank, cached) in self.cache_tags.iter().enumerate() {
            if *cached == Some(tag) {
                mask &= !(1_u64 << bank);
            }
            tag = tag.wrapping_add(self.geometry.bank_width);
        }
        mask << self.geometry.group_shift(access.address)
    }

    pub fn arbitrate(
        &mut self,
        requests: [Option<C220L1Access>; 3],
    ) -> [Option<C220L1Decision>; 3] {
        let mut decisions = [None; 3];
        let mut write_mask = 0;
        for port in C220L1Port::ALL {
            let index = port as usize;
            let Some(access) = requests[index] else {
                continue;
            };
            let mask = if port == C220L1Port::MteRead {
                self.read_bank_mask(access)
            } else {
                self.write_bank_mask(access)
            };
            let granted = write_mask & mask == 0;
            if granted {
                self.update_cache(access, mask, port == C220L1Port::MteRead);
                if port != C220L1Port::MteRead {
                    write_mask |= mask;
                }
            }
            decisions[index] = Some(C220L1Decision {
                access,
                bank_mask: mask,
                granted,
            });
        }
        decisions
    }

    fn update_cache(&mut self, access: C220L1Access, mask: u64, read: bool) {
        let mut tag = (access.address / u64::from(self.geometry.bank_width)) as u32;
        for (bank, cached) in self.cache_tags.iter_mut().enumerate() {
            if mask & (1_u64 << bank) != 0 {
                if read {
                    *cached = Some(tag);
                } else if *cached == Some(tag) {
                    *cached = None;
                }
            }
            tag = tag.wrapping_add(self.geometry.bank_width);
        }
    }
}
