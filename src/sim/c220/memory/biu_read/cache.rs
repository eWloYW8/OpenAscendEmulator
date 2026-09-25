use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220BiuReadCacheKind {
    Instruction,
    Data,
}

impl C220BiuReadCacheKind {
    fn index(self) -> usize {
        match self {
            Self::Instruction => 0,
            Self::Data => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220BiuReadCacheConfig {
    pub request_capacity: NonZeroU32,
    pub response_capacity: NonZeroU32,
    pub request_latency: u64,
    pub response_latency: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CachePort {
    config: C220BiuReadCacheConfig,
    input: VecDeque<C220MemoryReadCommand>,
    returns: VecDeque<C220MemoryReadReturn>,
}

fn endpoint(tag: C220MemoryReadId) -> Option<(usize, usize)> {
    match tag {
        C220MemoryReadId::InstructionCache { port, .. } => Some((0, port as usize)),
        C220MemoryReadId::DataCache { port, .. } => Some((1, port as usize)),
        C220MemoryReadId::Mte(_) => None,
    }
}

impl C220BiuBusReads {
    pub fn cache_can_send(&self, kind: C220BiuReadCacheKind, port: u32) -> bool {
        self.cache_ports[kind.index()]
            .get(port as usize)
            .is_some_and(|port| port.input.len() < port.config.request_capacity.get() as usize)
    }

    pub fn add_cache_port(
        &mut self,
        kind: C220BiuReadCacheKind,
        config: C220BiuReadCacheConfig,
    ) -> Result<u32, C220BiuBusReadError> {
        if !self.is_idle() {
            return Err(C220BiuBusReadError::InvalidPort);
        }
        let ports = &mut self.cache_ports[kind.index()];
        let index = u32::try_from(ports.len()).map_err(|_| C220BiuBusReadError::InvalidPort)?;
        ports.push(CachePort {
            config,
            input: VecDeque::new(),
            returns: VecDeque::new(),
        });
        self.last_cache[kind.index()] = None;
        Ok(index)
    }

    pub fn send_cache_command(
        &mut self,
        tick: u64,
        mut command: C220MemoryReadCommand,
    ) -> Result<bool, C220BiuBusReadError> {
        let (kind, index) = endpoint(command.tag).ok_or(C220BiuBusReadError::InvalidPort)?;
        let port = self.cache_ports[kind]
            .get_mut(index)
            .ok_or(C220BiuBusReadError::InvalidPort)?;
        if command.bytes == 0 || self.transactions.contains_key(&command.tag) {
            return Err(C220BiuBusReadError::InvalidRequest(command.tag));
        }
        if port.input.len() >= port.config.request_capacity.get() as usize {
            return Ok(false);
        }
        command.ready_tick = tick
            .checked_add(port.config.request_latency)
            .ok_or(C220BiuBusReadError::TimeOverflow)?;
        self.transactions.insert(
            command.tag,
            Transaction {
                expected: command.bytes.div_ceil(128),
                sent: false,
                received: BTreeSet::new(),
                delivered: 0,
            },
        );
        port.input.push_back(command);
        Ok(true)
    }

    pub fn cache_returns(
        &self,
        kind: C220BiuReadCacheKind,
        port: u32,
    ) -> Option<&VecDeque<C220MemoryReadReturn>> {
        Some(&self.cache_ports[kind.index()].get(port as usize)?.returns)
    }

    pub fn take_cache_return(
        &mut self,
        tick: u64,
        kind: C220BiuReadCacheKind,
        port: u32,
    ) -> Option<C220MemoryReadBeat> {
        let response = self.cache_ports[kind.index()]
            .get_mut(port as usize)?
            .returns
            .pop_front_if(|head| head.ready_tick <= tick)?;
        let transaction = self
            .transactions
            .get_mut(&response.beat.tag)
            .expect("pending cache read");
        transaction.delivered += 1;
        if transaction.delivered == transaction.expected {
            self.transactions.remove(&response.beat.tag);
        }
        Some(response.beat)
    }

    pub(super) fn select_cache(&mut self, tick: u64) -> Option<C220MemoryReadCommand> {
        for kind in 0..2 {
            let count = self.cache_ports[kind].len();
            if count == 0 {
                continue;
            }
            let start = self.last_cache[kind].map_or(0, |last| (last + 1) % count);
            for offset in 0..count {
                let index = (start + offset) % count;
                if let Some(command) = self.cache_ports[kind][index]
                    .input
                    .pop_front_if(|head| head.ready_tick <= tick)
                {
                    self.last_cache[kind] = Some(index);
                    return Some(command);
                }
            }
        }
        None
    }

    pub(super) fn forward_cache_return(
        &mut self,
        tick: u64,
        bus_port: usize,
    ) -> Result<(), C220BiuBusReadError> {
        let Some(mut response) = self.returns[bus_port]
            .front()
            .copied()
            .filter(|head| head.ready_tick <= tick)
        else {
            return Ok(());
        };
        let (kind, index) = endpoint(response.beat.tag).ok_or(C220BiuBusReadError::InvalidPort)?;
        let port = self.cache_ports[kind]
            .get_mut(index)
            .ok_or(C220BiuBusReadError::InvalidPort)?;
        if port.returns.len() >= port.config.response_capacity.get() as usize {
            return Ok(());
        }
        response.ready_tick = tick
            .checked_add(port.config.response_latency)
            .ok_or(C220BiuBusReadError::TimeOverflow)?;
        port.returns.push_back(response);
        self.returns[bus_port].pop_front();
        self.outstanding = self.outstanding.wrapping_sub(1);
        Ok(())
    }
}
