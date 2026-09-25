use super::store_buffer::C220LsuMemory;

mod data;
pub use data::{C220CacheLocation, C220CacheRefill, C220DataCache};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220AtomicCacheHit {
    ReplacedCleanLine,
    UpdatedDirtyAtomicLine,
    DirtyNonAtomicConflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CacheTag {
    pub atomic: bool,
    pub valid: bool,
    pub dirty: bool,
    pub age: u32,
    pub memory: C220LsuMemory,
    pub tag: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CacheMaintenanceTarget {
    All,
    Ub,
    External,
    Atomic,
}

impl C220CacheMaintenanceTarget {
    pub const fn matches(self, tag: C220CacheTag) -> bool {
        match self {
            Self::All => true,
            Self::Ub => matches!(tag.memory, C220LsuMemory::Ub),
            Self::External => matches!(tag.memory, C220LsuMemory::External),
            Self::Atomic => tag.atomic,
        }
    }

    pub const fn cleans(self, tag: C220CacheTag) -> bool {
        tag.dirty && (!tag.atomic || matches!(self, Self::All | Self::Atomic))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CacheWriteback {
    pub address: u64,
    pub memory: C220LsuMemory,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220CacheError {
    #[error("cache must have at least one way, with its stack partition within the way count")]
    InvalidPartition,
    #[error("cache way is out of range")]
    InvalidWay,
    #[error("cache index or tag shift exceeds the address width")]
    InvalidShift,
    #[error("cache geometry is empty, inconsistent, or cannot cover the index mask")]
    InvalidGeometry,
    #[error("cache line has the wrong byte length")]
    InvalidLineSize,
    #[error("cache index is out of range")]
    InvalidIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CacheAddressLayout {
    index_shift: u32,
    index_mask: u32,
    tag_shift: u32,
    tag_mask: u64,
}

impl C220CacheAddressLayout {
    pub fn new(
        index_shift: u32,
        index_mask: u32,
        tag_shift: u32,
        tag_mask: u64,
    ) -> Result<Self, C220CacheError> {
        if index_shift >= 64 || tag_shift >= 64 {
            return Err(C220CacheError::InvalidShift);
        }
        Ok(Self {
            index_shift,
            index_mask,
            tag_shift,
            tag_mask,
        })
    }

    pub fn index(self, address: u64) -> u32 {
        ((address >> self.index_shift) as u32) & self.index_mask
    }

    pub fn tag(self, address: u64) -> u64 {
        (address >> self.tag_shift) & self.tag_mask
    }

    pub fn line_address(self, index: u32, tag: u64) -> u64 {
        tag.wrapping_shl(self.tag_shift) | u64::from(index.wrapping_shl(self.index_shift))
    }
}

/// Per-index tag state. Cache geometry and initial tags are supplied by the
/// selected device profile. Data movement and dirty eviction are caller-owned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CacheSet {
    ways: Vec<C220CacheTag>,
    stack_ways: Option<usize>,
}

impl C220CacheSet {
    pub fn new(ways: Vec<C220CacheTag>, stack_ways: Option<usize>) -> Result<Self, C220CacheError> {
        if ways.is_empty() || stack_ways.is_some_and(|count| count > ways.len()) {
            return Err(C220CacheError::InvalidPartition);
        }
        Ok(Self { ways, stack_ways })
    }

    pub fn ways(&self) -> &[C220CacheTag] {
        &self.ways
    }

    /// Preserve a dirty victim's bytes before its storage is reused. The caller
    /// retains this writeback until the lower-level write completes. Invalidation
    /// leaves the tag, dirty flag and replacement counter unchanged.
    pub fn evict_dirty(
        &mut self,
        way: usize,
        index: u32,
        layout: C220CacheAddressLayout,
        bytes: &[u8],
    ) -> Result<Option<C220CacheWriteback>, C220CacheError> {
        let entry = self.ways.get_mut(way).ok_or(C220CacheError::InvalidWay)?;
        if !entry.valid || !entry.dirty {
            return Ok(None);
        }
        let writeback = C220CacheWriteback {
            address: layout.line_address(index, entry.tag),
            memory: entry.memory,
            bytes: bytes.to_vec(),
        };
        entry.valid = false;
        Ok(Some(writeback))
    }

    fn partition(&self, address: u64) -> std::ops::Range<usize> {
        match self.stack_ways {
            Some(count) if address >> 63 != 0 => 0..count,
            Some(count) => count..self.ways.len(),
            None => 0..self.ways.len(),
        }
    }

    /// Locate a matching tag independently of its valid bit; response handlers
    /// use this operation separately from demand-access hit lookup.
    pub fn find_way(&self, address: u64, tag: u64, memory: C220LsuMemory) -> Option<usize> {
        self.partition(address)
            .find(|&way| self.ways[way].tag == tag && self.ways[way].memory == memory)
    }

    /// Ordinary UB/external demand access. Cache-maintenance matching is separate.
    pub fn lookup(&mut self, address: u64, tag: u64, memory: C220LsuMemory) -> Option<usize> {
        let way = self.partition(address).find(|&way| {
            let entry = self.ways[way];
            entry.valid && entry.tag == tag && entry.memory == memory
        })?;
        self.promote(way).expect("matched cache way exists");
        Some(way)
    }

    pub fn promote(&mut self, way: usize) -> Result<(), C220CacheError> {
        if way >= self.ways.len() {
            return Err(C220CacheError::InvalidWay);
        }
        let age = self
            .ways
            .iter()
            .enumerate()
            .filter(|(index, entry)| *index != way && entry.valid)
            .map(|(_, entry)| entry.age)
            .max()
            .unwrap_or(0)
            .wrapping_add(1);
        self.ways[way].age = age;
        Ok(())
    }

    /// Select the first invalid way, otherwise use the replacement counters.
    /// Selection starts with way zero and age zero; a nonzero partition start
    /// does not initialize those values from that partition's first valid way.
    pub fn victim(&self, address: u64) -> usize {
        let mut selected = 0;
        let mut minimum = 0;
        for way in self.partition(address) {
            let entry = self.ways[way];
            if !entry.valid {
                return way;
            }
            if way == 0 || entry.age < minimum {
                minimum = entry.age;
                selected = way;
            }
        }
        selected
    }

    /// Install metadata after the caller has handled any dirty victim. Refill
    /// increments this way's counter instead of performing a hit promotion.
    pub fn install(
        &mut self,
        way: usize,
        tag: u64,
        memory: C220LsuMemory,
        dirty: bool,
    ) -> Result<C220CacheTag, C220CacheError> {
        let entry = self.ways.get_mut(way).ok_or(C220CacheError::InvalidWay)?;
        let previous = *entry;
        *entry = C220CacheTag {
            atomic: false,
            valid: true,
            dirty,
            age: entry.age.wrapping_add(1),
            memory,
            tag,
        };
        Ok(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_partition_and_hit_counters_have_distinct_rules() {
        let empty = C220CacheTag {
            atomic: false,
            valid: false,
            dirty: false,
            age: 0,
            memory: C220LsuMemory::Ub,
            tag: 0,
        };
        let mut set = C220CacheSet::new(vec![empty; 4], Some(2)).unwrap();
        assert_eq!(set.victim(0), 2);
        assert_eq!(set.victim(1 << 63), 0);
        assert_eq!(set.find_way(0, 0, C220LsuMemory::Ub), Some(2));
        assert_eq!(set.lookup(0, 0, C220LsuMemory::Ub), None);
        for way in 0..4 {
            set.install(way, 7, C220LsuMemory::External, false).unwrap();
        }
        assert_eq!(set.victim(0), 0);
        assert_eq!(set.lookup(0, 7, C220LsuMemory::Ub), None);
        assert_eq!(set.lookup(0, 7, C220LsuMemory::External), Some(2));
        assert_eq!(set.ways()[2].age, 2);
        set.install(0, 8, C220LsuMemory::Ub, true).unwrap();
        assert_eq!(set.ways()[0].age, 2);
        assert_eq!(set.victim(1 << 63), 1);
        let mut ways = set.ways().to_vec();
        ways[0].age = u32::MAX;
        let mut set = C220CacheSet::new(ways, None).unwrap();
        set.promote(2).unwrap();
        assert_eq!(set.ways()[2].age, 0);
        assert_eq!(set.victim(0), 2);
        let layout = C220CacheAddressLayout::new(6, 0x3f, 12, 0xff).unwrap();
        assert_eq!(layout.index(0xabc0), 0x2f);
        assert_eq!(layout.tag(0xabc0), 0xa);
        assert_eq!(layout.line_address(0x2f, 0xa), 0xabc0);
        let mut bytes = [0x55; 64];
        let previous = set.ways()[0];
        let writeback = set.evict_dirty(0, 3, layout, &bytes).unwrap().unwrap();
        bytes.fill(0xaa);
        assert_eq!(writeback.bytes, [0x55; 64]);
        assert_eq!(writeback.address, layout.line_address(3, previous.tag));
        assert!(!set.ways()[0].valid);
        assert!(set.ways()[0].dirty);
        assert_eq!(set.ways()[0].age, previous.age);
        assert!(set.evict_dirty(0, 3, layout, &bytes).unwrap().is_none());
    }
}
