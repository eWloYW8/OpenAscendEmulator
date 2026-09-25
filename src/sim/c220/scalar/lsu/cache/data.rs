use super::{
    C220CacheAddressLayout, C220CacheError, C220CacheSet, C220CacheWriteback, C220LsuMemory,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CacheLocation {
    pub index: u32,
    pub way: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CacheRefill {
    pub location: C220CacheLocation,
    pub writeback: Option<C220CacheWriteback>,
}

/// Tag and data RAM with explicit geometry, independent of response transport.
/// Dirty evictions return owned writebacks that the controller must enqueue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220DataCache {
    layout: C220CacheAddressLayout,
    line_bytes: usize,
    sets: Vec<C220CacheSet>,
    data: Vec<Vec<Vec<u8>>>,
}

impl C220DataCache {
    /// Construct zero-filled data RAM using caller-supplied initial tag state.
    pub fn new(
        layout: C220CacheAddressLayout,
        line_bytes: usize,
        sets: Vec<C220CacheSet>,
    ) -> Result<Self, C220CacheError> {
        let Some(first) = sets.first() else {
            return Err(C220CacheError::InvalidGeometry);
        };
        if line_bytes == 0
            || u64::from(layout.index_mask) >= sets.len() as u64
            || sets
                .iter()
                .any(|set| set.ways.len() != first.ways.len() || set.stack_ways != first.stack_ways)
        {
            return Err(C220CacheError::InvalidGeometry);
        }
        let data = vec![vec![vec![0; line_bytes]; first.ways.len()]; sets.len()];
        Ok(Self {
            layout,
            line_bytes,
            sets,
            data,
        })
    }

    pub const fn line_bytes(&self) -> usize {
        self.line_bytes
    }

    pub fn sets(&self) -> &[C220CacheSet] {
        &self.sets
    }

    pub fn replacement_requires_writeback(&self, address: u64, partition_address: u64) -> bool {
        let set = &self.sets[self.layout.index(address) as usize];
        let tag = set.ways[set.victim(partition_address)];
        tag.valid && tag.dirty
    }

    pub fn lookup(&mut self, address: u64, memory: C220LsuMemory) -> Option<C220CacheLocation> {
        self.lookup_partitioned(address, address, memory)
    }

    pub fn lookup_partitioned(
        &mut self,
        address: u64,
        partition_address: u64,
        memory: C220LsuMemory,
    ) -> Option<C220CacheLocation> {
        let index = self.layout.index(address);
        let way = self.sets[index as usize].lookup(
            partition_address,
            self.layout.tag(address),
            memory,
        )?;
        Some(C220CacheLocation { index, way })
    }

    pub fn find_way(&self, address: u64, memory: C220LsuMemory) -> Option<C220CacheLocation> {
        let index = self.layout.index(address);
        let way = self.sets[index as usize].find_way(address, self.layout.tag(address), memory)?;
        Some(C220CacheLocation { index, way })
    }

    pub fn line(&self, location: C220CacheLocation) -> Result<&[u8], C220CacheError> {
        self.data
            .get(location.index as usize)
            .ok_or(C220CacheError::InvalidIndex)?
            .get(location.way)
            .map(Vec::as_slice)
            .ok_or(C220CacheError::InvalidWay)
    }

    pub fn line_mut(&mut self, location: C220CacheLocation) -> Result<&mut [u8], C220CacheError> {
        self.data
            .get_mut(location.index as usize)
            .ok_or(C220CacheError::InvalidIndex)?
            .get_mut(location.way)
            .map(Vec::as_mut_slice)
            .ok_or(C220CacheError::InvalidWay)
    }

    pub fn mark_dirty(&mut self, location: C220CacheLocation) -> Result<(), C220CacheError> {
        self.sets
            .get_mut(location.index as usize)
            .ok_or(C220CacheError::InvalidIndex)?
            .ways
            .get_mut(location.way)
            .ok_or(C220CacheError::InvalidWay)?
            .dirty = true;
        Ok(())
    }

    /// `address` selects tag/index; `partition_address` retains the address used
    /// to distinguish stack traffic before any lower-level address translation.
    pub fn refill(
        &mut self,
        address: u64,
        partition_address: u64,
        memory: C220LsuMemory,
        bytes: &[u8],
        dirty: bool,
    ) -> Result<C220CacheRefill, C220CacheError> {
        if bytes.len() != self.line_bytes {
            return Err(C220CacheError::InvalidLineSize);
        }
        let index = self.layout.index(address);
        let set = &mut self.sets[index as usize];
        let way = set.victim(partition_address);
        let line = &mut self.data[index as usize][way];
        let writeback = set.evict_dirty(way, index, self.layout, line)?;
        set.install(way, self.layout.tag(address), memory, dirty)?;
        line.copy_from_slice(bytes);
        Ok(C220CacheRefill {
            location: C220CacheLocation { index, way },
            writeback,
        })
    }
}
