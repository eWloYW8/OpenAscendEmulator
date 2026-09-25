use super::super::cache::{C220CacheRefill, C220DataCache};
use super::super::miss_buffer::C220LsuMissReadAction;
use super::super::read_queue::{
    C220LsuReadError, C220LsuReadId, C220LsuReadOwner, C220LsuReadState,
};
use super::super::store_buffer::{C220LsuLineKey, C220LsuStoreState};
use super::{
    C220LsuReadCompletion, C220LsuRequestId, C220LsuRequestScheduler, C220LsuSchedulerError,
};

impl C220LsuRequestScheduler {
    /// Join an existing fetch or create a read owned by the new miss entry.
    pub fn enqueue_load_miss(
        &mut self,
        line: C220LsuLineKey,
        request: C220LsuRequestId,
        partition_address: u64,
    ) -> Result<Option<C220LsuReadId>, C220LsuSchedulerError> {
        self.reads.check_enqueue()?;
        let action = self.misses.push(line, request, &mut self.stores)?;
        if action == C220LsuMissReadAction::IssueRead {
            Ok(Some(self.reads.enqueue(
                line,
                partition_address,
                C220LsuReadOwner::Miss,
            )?))
        } else {
            Ok(None)
        }
    }

    /// Start the lower-level fetch selected by the store-buffer controller.
    pub fn enqueue_store_read(
        &mut self,
        line: C220LsuLineKey,
        partition_address: u64,
    ) -> Result<C220LsuReadId, C220LsuSchedulerError> {
        self.reads.check_enqueue()?;
        if self
            .stores
            .entry(line)
            .is_none_or(|entry| entry.state() != C220LsuStoreState::Idle)
        {
            return Err(C220LsuSchedulerError::UnexpectedStoreResponse);
        }
        self.stores.set_state(line, C220LsuStoreState::Fetching)?;
        Ok(self
            .reads
            .enqueue(line, partition_address, C220LsuReadOwner::Store)?)
    }

    /// Read backing storage at response time, then refill and notify the
    /// initiating buffer. Failed storage reads retain ownership for retry.
    pub fn apply_cached_read_response<E>(
        &mut self,
        id: C220LsuReadId,
        cache: &mut C220DataCache,
        read: impl FnOnce(C220LsuLineKey, usize) -> Result<Vec<u8>, E>,
    ) -> Result<(C220LsuReadCompletion, C220CacheRefill), E>
    where
        E: From<C220LsuSchedulerError>,
    {
        let request = *self.reads.request(id).ok_or(C220LsuSchedulerError::Read(
            C220LsuReadError::MissingRequest,
        ))?;
        match request.state {
            C220LsuReadState::InFlight => {
                self.reads
                    .begin_response(id)
                    .map_err(C220LsuSchedulerError::from)?;
            }
            C220LsuReadState::Responding => {}
            C220LsuReadState::Queued => {
                return Err(C220LsuSchedulerError::Read(C220LsuReadError::InvalidState).into());
            }
        }
        let bytes = read(request.line, request.byte_len)?;
        let result = match request.owner {
            C220LsuReadOwner::Miss => self.complete_cached_miss_read(
                request.line,
                request.partition_address,
                &bytes,
                cache,
            ),
            C220LsuReadOwner::Store => self.complete_cached_store_read(
                request.line,
                request.partition_address,
                &bytes,
                cache,
            ),
        }?;
        self.reads
            .finish_response(id)
            .map_err(C220LsuSchedulerError::from)?;
        Ok(result)
    }
}
