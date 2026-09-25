use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CachePort {
    pub input: VecDeque<C220MemoryWriteCommand>,
    pub capacities: [usize; 2],
    pub returns: [VecDeque<C220MemoryWriteTransfer>; 2],
}

impl C220BiuBusWrites {
    pub fn cache_can_send(&self, port: u32) -> bool {
        self.cache_ports
            .get(port as usize)
            .is_some_and(|port| port.input.len() < 2)
    }

    /// Accept a cache endpoint send into its two-slot, one-tick ingress.
    pub fn send_cache_command(
        &mut self,
        tick: u64,
        mut command: C220MemoryWriteCommand,
    ) -> Result<bool, C220BiuBusWriteError> {
        let C220MemoryWriteId::Cache { port, .. } = command.tag else {
            return Err(C220BiuBusWriteError::InvalidPort);
        };
        if self.cache_ports.get(port as usize).is_none() {
            return Err(C220BiuBusWriteError::InvalidPort);
        }
        if self.phases.contains_key(&command.tag)
            || self
                .cache_ports
                .iter()
                .any(|port| port.input.iter().any(|head| head.tag == command.tag))
        {
            return Err(C220BiuBusWriteError::InvalidPhase(command.tag));
        }
        if !self.cache_can_send(port) {
            return Ok(false);
        }
        command.ready_tick = next_tick(tick)?;
        self.cache_ports[port as usize].input.push_back(command);
        Ok(true)
    }

    pub(crate) fn admit_inputs(
        &mut self,
        tick: u64,
        mte: Option<C220MemoryWriteCommand>,
    ) -> Result<Option<C220MemoryWriteId>, C220BiuBusWriteError> {
        let heads: Vec<_> = self
            .cache_ports
            .iter()
            .map(|port| port.input.front().copied())
            .collect();
        let selected = self.admit_commands(tick, &heads, mte)?;
        if let Some(C220MemoryWriteId::Cache { port, .. }) = selected {
            self.cache_ports[port as usize].input.pop_front();
        }
        Ok(selected)
    }

    /// The shared data-receive process scans every cache queue before MTE.
    /// A valid MTE input may wake it even before a cache head has aged.
    pub(crate) fn advance_cache_data(
        &mut self,
        tick: u64,
        mte_ready: bool,
    ) -> Result<(), C220BiuBusWriteError> {
        let ready = mte_ready
            || self.cache_ports.iter().any(|port| {
                port.returns[0]
                    .front()
                    .is_some_and(|head| head.ready_tick <= tick)
            });
        if !ready || self.last_cache_data == Some(tick) {
            return Ok(());
        }
        let ready_tick = next_tick(tick)?;
        self.last_cache_data = Some(tick);
        for port in &mut self.cache_ports {
            if let Some(mut response) = port.returns[0].pop_front() {
                response.ready_tick = ready_tick;
                self.phases.insert(response.tag, Phase::DataQueued);
                self.data.push_back(response);
            }
        }
        Ok(())
    }

    /// Configure response queue capacities before admitting traffic.
    pub fn add_cache_port(&mut self, capacities: [usize; 2]) -> Result<u32, C220BiuBusWriteError> {
        if !self.is_idle() || capacities.contains(&0) {
            return Err(C220BiuBusWriteError::InvalidPort);
        }
        let port =
            u32::try_from(self.cache_ports.len()).map_err(|_| C220BiuBusWriteError::InvalidPort)?;
        self.cache_ports.push(CachePort {
            input: VecDeque::new(),
            capacities,
            returns: Default::default(),
        });
        Ok(port)
    }

    /// Sample ready endpoint heads once per process tick. Cache ports have
    /// round-robin priority ahead of MTE. Admission does not reserve bus credit.
    /// The returned ID identifies the one endpoint head the caller must consume.
    pub fn admit_commands(
        &mut self,
        tick: u64,
        caches: &[Option<C220MemoryWriteCommand>],
        mte: Option<C220MemoryWriteCommand>,
    ) -> Result<Option<C220MemoryWriteId>, C220BiuBusWriteError> {
        if caches.len() != self.cache_ports.len()
            || mte.is_some_and(|command| !matches!(command.tag, C220MemoryWriteId::Mte(_)))
        {
            return Err(C220BiuBusWriteError::InvalidPort);
        }
        for (port, command) in caches.iter().enumerate() {
            if command.is_some_and(|command| !matches!(command.tag, C220MemoryWriteId::Cache { port: source, .. } if source as usize == port)) {
                return Err(C220BiuBusWriteError::InvalidPort);
            }
        }
        if self.last_admission == Some(tick) || !self.can_receive_command() {
            return Ok(None);
        }
        let selected = (0..caches.len())
            .map(|offset| (self.next_cache + offset) % caches.len())
            .find(|&port| caches[port].is_some_and(|command| command.ready_tick <= tick));
        let command = selected
            .and_then(|port| caches[port])
            .or_else(|| mte.filter(|command| command.ready_tick <= tick));
        if let Some(command) = command {
            self.push_command(tick, command)?;
            if let Some(port) = selected {
                self.next_cache = (port + 1) % caches.len();
            }
        }
        self.last_admission = Some(tick);
        Ok(command.map(|command| command.tag))
    }

    pub fn cache_returns(
        &self,
        port: u32,
        kind: C220BiuWriteReturnKind,
    ) -> Option<&VecDeque<C220MemoryWriteTransfer>> {
        self.cache_ports
            .get(port as usize)
            .map(|port| &port.returns[kind.index()])
    }

    /// Transfer a cache response to its owner. DBID consumption makes its data
    /// eligible for the shared data path; it does not release outstanding credit.
    pub fn take_cache_return(
        &mut self,
        tick: u64,
        port: u32,
        kind: C220BiuWriteReturnKind,
    ) -> Option<C220MemoryWriteId> {
        let response = self.cache_ports.get_mut(port as usize)?.returns[kind.index()]
            .pop_front_if(|head| head.ready_tick <= tick)?;
        match kind {
            C220BiuWriteReturnKind::Dbid => {
                self.phases.insert(response.tag, Phase::DbidDelivered);
            }
            C220BiuWriteReturnKind::Completion => {
                self.phases.remove(&response.tag);
            }
        }
        Some(response.tag)
    }
}
