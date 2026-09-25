const ADDRESS_ROOT_MASK: u64 = 0x1fffffe000000;
const LOCAL_OFFSET_MASK: u64 = 0x7ffff;
const EXTERNAL_ADDRESS_MASK: u64 = 0xffffffffffff;

pub(crate) const C220_UB_BYTES: u64 = 0x30000;

/// Optional fixed bases. Unconfigured bases are read from SPR67 and SPR68.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220ScalarAddressConfig {
    pub system_base: Option<u64>,
    pub stack_base: Option<u64>,
}

impl C220ScalarAddressConfig {
    /// The LSU selects each base independently.
    pub fn lsu_roots(self, spr67: Option<u64>, spr68: Option<u64>) -> Option<(u64, u64)> {
        self.system_base.or(spr67).zip(self.stack_base.or(spr68))
    }

    /// Functional scalar execution uses fixed bases only when both are configured.
    pub fn functional_roots(self, spr67: Option<u64>, spr68: Option<u64>) -> Option<(u64, u64)> {
        self.system_base
            .zip(self.stack_base)
            .or_else(|| spr67.zip(spr68))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ScalarMappedAddress {
    pub address: u64,
    pub memory: super::lsu::store_buffer::C220LsuMemory,
    pub stack: bool,
}

impl C220ScalarMappedAddress {
    pub fn decode(address: u64, spr67: u64, spr68: u64) -> Option<Self> {
        use super::lsu::store_buffer::C220LsuMemory;
        let root = spr67 & ADDRESS_ROOT_MASK;
        let (address, memory, stack) =
            if address & 0x1000000 != 0 || address & ADDRESS_ROOT_MASK != root {
                (
                    address & EXTERNAL_ADDRESS_MASK,
                    C220LsuMemory::External,
                    false,
                )
            } else {
                let bank = (address >> 20) & 0x1f;
                if bank == 0 && address & 0x80000 != 0 {
                    (address & LOCAL_OFFSET_MASK, C220LsuMemory::Ub, false)
                } else if bank & 0xf == 0 {
                    return None;
                } else if root == spr68 & ADDRESS_ROOT_MASK {
                    (address & LOCAL_OFFSET_MASK, C220LsuMemory::Ub, true)
                } else {
                    (
                        (spr68 & EXTERNAL_ADDRESS_MASK)
                            .wrapping_sub(0x100000)
                            .wrapping_add(address & 0xffffff),
                        C220LsuMemory::External,
                        true,
                    )
                }
            };
        Some(Self {
            address,
            memory,
            stack,
        })
    }

    pub const fn cache_address(self, partition_stack: bool) -> u64 {
        if self.stack && partition_stack {
            self.address | (1 << 63)
        } else {
            self.address
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ScalarRoute {
    Hbm(u64),
    Ub(u64),
    Unsupported,
}

pub(crate) fn classify_c220_scalar_address(
    address: u64,
    spr67: u64,
    spr68: u64,
) -> C220ScalarRoute {
    use super::lsu::store_buffer::C220LsuMemory;
    match C220ScalarMappedAddress::decode(address, spr67, spr68) {
        Some(mapped) if mapped.memory == C220LsuMemory::External => {
            C220ScalarRoute::Hbm(mapped.address)
        }
        Some(mapped) if !mapped.stack || mapped.address < C220_UB_BYTES => {
            C220ScalarRoute::Ub(mapped.address)
        }
        _ => C220ScalarRoute::Unsupported,
    }
}
