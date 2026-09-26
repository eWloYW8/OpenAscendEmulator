use std::collections::VecDeque;

use thiserror::Error;

use crate::isa::c220::cube::{
    C220_CUBE_ARRAY_EDGE, C220CubeDataType, C220CubeGeometry, C220CubeInstruction,
    C220CubeOperation, C220MmadParameters,
};
use crate::sim::c220::cube::control::{C220CubeIssueDelay, C220CubeTimingControl, C220F32MmadMode};
use crate::sim::c220::cube::uop::{C220CubeUop, C220CubeUopRelease};
use crate::sim::c220::memory::{
    C220L0c, C220L0cError, C220L0cMaster, C220L0cUnitFlagBlock, C220L0cWritePortBlock,
};
use crate::sim::c220::sync::{
    C220CubeFlagCheckpoint, C220CubeFlagStages, C220HardwareFlagState, C220HardwareFlagTimingError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeFsmVersion {
    V0,
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeV1FrameOrder {
    NThenM,
    MThenN,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeConfig {
    pub array_edge: u16,
    pub cube_spec_npe: u16,
    pub cube_stage_num: u16,
    pub fsm_version: C220CubeFsmVersion,
    pub v1_n2_mode: bool,
    pub v1_m_priority: bool,
    pub flag_stages: C220CubeFlagStages,
}

impl Default for C220CubeConfig {
    fn default() -> Self {
        Self {
            array_edge: C220_CUBE_ARRAY_EDGE,
            cube_spec_npe: 256,
            cube_stage_num: 22,
            fsm_version: C220CubeFsmVersion::V1,
            v1_n2_mode: false,
            v1_m_priority: false,
            flag_stages: C220CubeFlagStages::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeTicket {
    /// Reception by the execution pipeline, after the frontend queue.
    pub accept_tick: u64,
    pub first_uop_tick: Option<u64>,
    pub last_uop_tick: Option<u64>,
    pub retire_tick: u64,
    pub geometry: C220CubeGeometry,
    pub uop_count: u64,
    pub fsm_bubbles: u64,
    pub sparse_bubbles: u64,
    pub resource_wait_ticks: u64,
    pub resource_waits: C220CubeResourceWaits,
    pub issue_delay_wait_ticks: u64,
    pub issue_delay: C220CubeIssueDelay,
    pub fsm_version: C220CubeFsmVersion,
    pub v1_n2_mode: bool,
    pub v1_frame_order: C220CubeV1FrameOrder,
    pub v1_dtype_bubbles_per_uop: u8,
    pub bias_checkpoint_uop: Option<u64>,
    pub unit_flag_sync: bool,
}

/// Mutually exclusive reasons for cycles added by runtime backpressure.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CubeResourceWaits {
    pub instruction_order: u64,
    pub hardware_flag: u64,
    pub l0c_write_port: u64,
    pub l0c_unit_flag: u64,
}

enum ResourceWait {
    InstructionOrder,
    HardwareFlag,
    L0cWritePort,
    L0cUnitFlag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeL0cStallReason {
    WritePort(C220L0cWritePortBlock),
    UnitFlag(C220L0cUnitFlagBlock),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeL0cStall {
    pub instruction_id: u64,
    pub uop_id: u64,
    pub tick: u64,
    pub reason: C220CubeL0cStallReason,
}

impl C220CubeTicket {
    fn v1_shape_bubbles_through(self, uop_id: u64) -> Option<u64> {
        if self.geometry.m_tiles == 1 && self.geometry.n_tiles == 1 {
            return uop_id.checked_add(1);
        }
        if !self.v1_n2_mode
            || self.geometry.m_tiles.is_multiple_of(2)
            || self.geometry.n_tiles.is_multiple_of(2)
        {
            return Some(0);
        }
        let tail_start = self
            .uop_count
            .checked_sub(u64::from(self.geometry.k_tiles))?;
        Some(uop_id.checked_add(1)?.saturating_sub(tail_start))
    }

    pub fn planned_uop_tick(self, uop_id: u64) -> Option<u64> {
        if uop_id >= self.uop_count {
            return None;
        }
        let preceding_fsm_bubbles = match self.fsm_version {
            C220CubeFsmVersion::V0 => {
                let tail_start = self
                    .uop_count
                    .saturating_sub(u64::from(self.geometry.k_tiles));
                if self.fsm_bubbles != 0 && uop_id > tail_start {
                    uop_id - tail_start
                } else {
                    0
                }
            }
            C220CubeFsmVersion::V1 => uop_id
                .checked_add(1)?
                .checked_mul(u64::from(self.v1_dtype_bubbles_per_uop))?
                .checked_add(self.v1_shape_bubbles_through(uop_id)?)?,
        };
        self.accept_tick
            .checked_add(1)?
            .checked_add(self.sparse_bubbles)?
            .checked_add(uop_id)?
            .checked_add(preceding_fsm_bubbles)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C220CubeTimingError {
    #[error("C220 Cube array edge must be 16, got {0}")]
    InvalidArrayEdge(u16),
    #[error("C220 Cube NPE count must be a nonzero divisor of 256, got {0}")]
    InvalidNpe(u16),
    #[error("C220 Cube V1 requires 256 processing elements, got {0}")]
    InvalidV1Npe(u16),
    #[error("C220 Cube stage count must be at least one")]
    InvalidStageCount,
    #[error("C220 Cube cannot accept an instruction until tick {ready_tick}")]
    Busy { ready_tick: u64 },
    #[error("C220 Cube timing computation overflowed")]
    TimeOverflow,
    #[error("C220 Cube ticket does not match the current timing state")]
    TicketMismatch,
    #[error(transparent)]
    L0c(#[from] C220L0cError),
    #[error(transparent)]
    HardwareFlag(#[from] C220HardwareFlagTimingError),
    #[error("tick {requested} precedes the previously observed Cube tick {previous}")]
    TimeReversed { requested: u64, previous: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CubePipeline {
    config: C220CubeConfig,
    now: u64,
    next_accept_tick: u64,
    in_flight: VecDeque<C220CubeInFlight>,
    last_retirements: Vec<C220CubeTicket>,
    last_uop_releases: Vec<C220CubeUopRelease>,
    last_l0c_stalls: Vec<C220CubeL0cStall>,
    previous_last_uop_tick: u64,
    live_issue_delay: Option<C220CubeIssueDelay>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct C220CubeInFlight {
    instruction_id: u64,
    ticket: C220CubeTicket,
    pending_uops: VecDeque<C220CubeUop>,
    next_uop_tick: Option<u64>,
    l0c_port_granted: bool,
    first_observation_tick: Option<u64>,
    first_bubbles_remaining: u64,
}

impl C220CubePipeline {
    pub fn new(config: C220CubeConfig) -> Result<Self, C220CubeTimingError> {
        validate_config(config)?;
        Ok(Self {
            config,
            now: 0,
            next_accept_tick: 0,
            in_flight: VecDeque::new(),
            last_retirements: Vec::new(),
            last_uop_releases: Vec::new(),
            last_l0c_stalls: Vec::new(),
            previous_last_uop_tick: 0,
            live_issue_delay: None,
        })
    }

    pub const fn config(&self) -> C220CubeConfig {
        self.config
    }

    /// Applies to subsequent issue attempts; advance to the write tick before changing it.
    pub fn set_issue_delay(&mut self, control: C220CubeIssueDelay) {
        self.live_issue_delay = Some(control);
    }

    pub const fn next_accept_tick(&self) -> u64 {
        self.next_accept_tick
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.in_flight
            .iter()
            .map(|flight| flight.ticket.retire_tick)
            .max()
    }

    pub fn next_event_tick(&self) -> Option<u64> {
        let retirement = self.in_flight.front().and_then(|flight| {
            (flight.pending_uops.is_empty()
                && (flight.l0c_port_granted || flight.ticket.uop_count == 0))
                .then_some(flight.ticket.retire_tick.max(self.now))
        });
        self.in_flight
            .iter()
            .find_map(|flight| flight.next_uop_tick)
            .map(|tick| tick.max(self.now))
            .into_iter()
            .chain(retirement)
            .chain(
                self.in_flight
                    .iter()
                    .filter(|flight| flight.ticket.uop_count == 0)
                    .map(|flight| flight.ticket.retire_tick.max(self.now)),
            )
            .min()
    }

    pub fn pending_retirement_count(&self) -> usize {
        self.in_flight.len()
    }

    pub fn last_retirements(&self) -> &[C220CubeTicket] {
        &self.last_retirements
    }

    pub fn last_uop_releases(&self) -> &[C220CubeUopRelease] {
        &self.last_uop_releases
    }

    pub fn last_l0c_stalls(&self) -> &[C220CubeL0cStall] {
        &self.last_l0c_stalls
    }

    pub fn advance_to(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
        hardware_flags: &mut C220HardwareFlagState,
    ) -> Result<&[C220CubeTicket], C220CubeTimingError> {
        self.begin_advance();
        self.advance_in_batch_to(tick, l0c, hardware_flags)?;
        Ok(&self.last_retirements)
    }

    pub(crate) fn begin_advance(&mut self) {
        self.last_retirements.clear();
        self.last_uop_releases.clear();
        self.last_l0c_stalls.clear();
    }

    pub(crate) fn advance_in_batch_to(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
        hardware_flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220CubeTimingError> {
        if tick < self.now {
            return Err(C220CubeTimingError::TimeReversed {
                requested: tick,
                previous: self.now,
            });
        }
        while let Some(event_tick) = self
            .next_event_tick()
            .into_iter()
            .chain(hardware_flags.next_notification_tick())
            .min()
            .filter(|&next| next <= tick)
        {
            hardware_flags.advance_to(event_tick)?;
            self.advance_event_at(event_tick, l0c, hardware_flags)?;
            self.now = event_tick;
        }
        l0c.read_banks_mut().advance_to(tick)?;
        self.now = tick;
        Ok(())
    }

    fn advance_event_at(
        &mut self,
        tick: u64,
        l0c: &mut C220L0c,
        hardware_flags: &mut C220HardwareFlagState,
    ) -> Result<(), C220CubeTimingError> {
        for flight in &mut self.in_flight {
            if let Some(ready_tick) = flight.next_uop_tick {
                let ordered_tick = self
                    .previous_last_uop_tick
                    .checked_add(1)
                    .ok_or(C220CubeTimingError::TimeOverflow)?;
                if ready_tick < ordered_tick {
                    let uop = flight
                        .pending_uops
                        .front()
                        .expect("pending Cube tick has a uop");
                    delay_ticket_from_uop(
                        &mut flight.ticket,
                        uop.id,
                        ordered_tick - ready_tick,
                        ResourceWait::InstructionOrder,
                    )?;
                    flight.next_uop_tick = Some(ordered_tick);
                    self.next_accept_tick = self
                        .next_accept_tick
                        .max(next_instruction_tick(flight.ticket)?);
                }
            }
            while flight.next_uop_tick.is_some_and(|ready| ready <= tick) {
                let ready_tick = flight
                    .next_uop_tick
                    .expect("pending Cube uop has a ready tick");
                let uop = *flight
                    .pending_uops
                    .front()
                    .expect("pending Cube tick has a uop");
                l0c.scoreboard_mut().advance_to(ready_tick);

                if uop.id == 0
                    && let Some(resume_tick) =
                        hardware_flags.gate_cube_instruction(ready_tick, flight.instruction_id)?
                {
                    let delay = resume_tick
                        .checked_sub(ready_tick)
                        .ok_or(C220CubeTimingError::TimeOverflow)?;
                    delay_ticket_from_uop(
                        &mut flight.ticket,
                        uop.id,
                        delay,
                        ResourceWait::HardwareFlag,
                    )?;
                    flight.next_uop_tick = Some(resume_tick);
                    self.next_accept_tick = self
                        .next_accept_tick
                        .max(next_instruction_tick(flight.ticket)?);
                    continue;
                }

                if uop.id == 0 {
                    if flight.ticket.unit_flag_sync {
                        hardware_flags
                            .check_cube_unit_flag_set(flight.instruction_id, ready_tick)?;
                    }
                    let observed = *flight.first_observation_tick.get_or_insert(ready_tick);
                    let control = self.live_issue_delay.unwrap_or(flight.ticket.issue_delay);
                    let guard = self
                        .previous_last_uop_tick
                        .checked_add(u64::from(control.previous_issue_guard_ticks))
                        .ok_or(C220CubeTimingError::TimeOverflow)?;
                    let deadline = observed
                        .checked_add(u64::from(control.first_observation_delay_ticks))
                        .ok_or(C220CubeTimingError::TimeOverflow)?;
                    if control.enabled && ready_tick >= guard && ready_tick < deadline {
                        shift_ticket_from_uop(&mut flight.ticket, 0, 1)?;
                        flight.ticket.issue_delay_wait_ticks = flight
                            .ticket
                            .issue_delay_wait_ticks
                            .checked_add(1)
                            .ok_or(C220CubeTimingError::TimeOverflow)?;
                        flight.next_uop_tick = Some(
                            ready_tick
                                .checked_add(1)
                                .ok_or(C220CubeTimingError::TimeOverflow)?,
                        );
                        self.next_accept_tick = self
                            .next_accept_tick
                            .max(next_instruction_tick(flight.ticket)?);
                        continue;
                    }
                    if flight.first_bubbles_remaining != 0 {
                        flight.first_bubbles_remaining -= 1;
                        flight.next_uop_tick = Some(
                            ready_tick
                                .checked_add(1)
                                .ok_or(C220CubeTimingError::TimeOverflow)?,
                        );
                        continue;
                    }
                }

                let mut blocked = None;
                if uop.acquires_l0c_write_port && !flight.l0c_port_granted {
                    if let Some(reason) = l0c.write_arbiter().blocked_by(C220L0cMaster::Cube) {
                        blocked = Some(C220CubeL0cStallReason::WritePort(reason));
                    } else {
                        flight.l0c_port_granted =
                            l0c.write_arbiter_mut().grant(C220L0cMaster::Cube);
                    }
                }
                if blocked.is_none()
                    && let Some(request) = uop.l0c_write
                    && let Err(reason) = l0c.scoreboard_mut().admit(request.into(), ready_tick)?
                {
                    blocked = Some(C220CubeL0cStallReason::UnitFlag(reason));
                }
                if let Some(blocked) = blocked {
                    self.last_l0c_stalls.push(C220CubeL0cStall {
                        instruction_id: flight.instruction_id,
                        uop_id: uop.id,
                        tick: ready_tick,
                        reason: blocked,
                    });
                    let resume_tick = ready_tick
                        .checked_add(1)
                        .ok_or(C220CubeTimingError::TimeOverflow)?;
                    let reason = match blocked {
                        C220CubeL0cStallReason::WritePort(_) => ResourceWait::L0cWritePort,
                        C220CubeL0cStallReason::UnitFlag(_) => ResourceWait::L0cUnitFlag,
                    };
                    delay_ticket_from_uop(&mut flight.ticket, uop.id, 1, reason)?;
                    flight.next_uop_tick = Some(resume_tick);
                    self.next_accept_tick = self
                        .next_accept_tick
                        .max(next_instruction_tick(flight.ticket)?);
                    continue;
                }

                flight.pending_uops.pop_front();
                if flight.ticket.bias_checkpoint_uop == Some(uop.id) {
                    hardware_flags.enqueue_cube_checkpoint(
                        C220CubeFlagCheckpoint::Bias,
                        flight.instruction_id,
                        ready_tick,
                        self.config.flag_stages.bias,
                    )?;
                }
                if let Some(request) = uop.l0c_read {
                    l0c.read_banks_mut().send_cube_read(
                        ready_tick,
                        request.address,
                        request.bytes == 512,
                    )?;
                }
                self.last_uop_releases.push(C220CubeUopRelease {
                    instruction_id: flight.instruction_id,
                    uop,
                    issue_tick: ready_tick,
                });
                flight.next_uop_tick = if let Some(next) = flight.pending_uops.front() {
                    ready_tick.checked_add(1).and_then(|next_tick| {
                        next_tick.checked_add(u64::from(next.pre_issue_bubbles))
                    })
                } else {
                    None
                };
                if flight.pending_uops.is_empty() {
                    hardware_flags.enqueue_cube_checkpoint(
                        C220CubeFlagCheckpoint::ReadBuffers,
                        flight.instruction_id,
                        ready_tick,
                        self.config.flag_stages.read_buffers,
                    )?;
                    hardware_flags.enqueue_cube_checkpoint(
                        C220CubeFlagCheckpoint::WriteBuffer,
                        flight.instruction_id,
                        ready_tick,
                        self.config.flag_stages.write_buffer,
                    )?;
                    flight.ticket.last_uop_tick = Some(ready_tick);
                    self.previous_last_uop_tick = ready_tick;
                    self.next_accept_tick = self.next_accept_tick.max(ready_tick);
                } else if flight.next_uop_tick.is_none() {
                    return Err(C220CubeTimingError::TimeOverflow);
                }
            }
            if !flight.pending_uops.is_empty() {
                break;
            }
        }

        while let Some(index) = self
            .in_flight
            .iter()
            .enumerate()
            .find_map(|(index, flight)| {
                let ready = flight.ticket.retire_tick <= tick
                    && (flight.ticket.uop_count == 0
                        || (index == 0
                            && flight.pending_uops.is_empty()
                            && flight.l0c_port_granted));
                ready.then_some(index)
            })
        {
            let flight = self
                .in_flight
                .remove(index)
                .expect("completed Cube flight exists");
            if flight.l0c_port_granted {
                l0c.write_arbiter_mut().complete(C220L0cMaster::Cube);
            }
            self.last_retirements.push(flight.ticket);
        }
        Ok(())
    }

    pub fn preview_issue(
        &self,
        accept_tick: u64,
        instruction: C220CubeInstruction,
        parameters: C220MmadParameters,
        control: C220CubeTimingControl,
    ) -> Result<C220CubeTicket, C220CubeTimingError> {
        if accept_tick < self.next_accept_tick {
            return Err(C220CubeTimingError::Busy {
                ready_tick: self.next_accept_tick,
            });
        }
        schedule(
            self.config,
            accept_tick,
            instruction,
            parameters,
            control,
            self.previous_last_uop_tick,
        )
    }

    pub fn issue<I>(
        &mut self,
        mut ticket: C220CubeTicket,
        uops: I,
        instruction_id: u64,
        l0c: &mut C220L0c,
    ) -> Result<(), C220CubeTimingError>
    where
        I: IntoIterator<Item = C220CubeUop>,
    {
        if ticket.accept_tick < self.next_accept_tick {
            return Err(C220CubeTimingError::Busy {
                ready_tick: self.next_accept_tick,
            });
        }
        if ticket.accept_tick < self.now {
            return Err(C220CubeTimingError::TimeReversed {
                requested: ticket.accept_tick,
                previous: self.now,
            });
        }
        let pending_uops = uops.into_iter().collect::<VecDeque<_>>();
        if pending_uops.len() != usize::try_from(ticket.uop_count).unwrap_or(usize::MAX) {
            return Err(C220CubeTimingError::TicketMismatch);
        }
        let first_bubbles_remaining = pending_uops.front().map_or(0, |uop| {
            u64::from(uop.pre_issue_bubbles) + ticket.sparse_bubbles
        });
        let next_uop_tick = if ticket.uop_count == 0 {
            None
        } else {
            Some(
                ticket
                    .accept_tick
                    .checked_add(1)
                    .ok_or(C220CubeTimingError::TimeOverflow)?,
            )
        };
        let predicted_wait = ticket.issue_delay_wait_ticks;
        let remove_prediction = |tick: u64| {
            tick.checked_sub(predicted_wait)
                .ok_or(C220CubeTimingError::TicketMismatch)
        };
        ticket.first_uop_tick = ticket.first_uop_tick.map(remove_prediction).transpose()?;
        ticket.last_uop_tick = ticket.last_uop_tick.map(remove_prediction).transpose()?;
        ticket.retire_tick = ticket
            .retire_tick
            .checked_sub(predicted_wait)
            .ok_or(C220CubeTimingError::TicketMismatch)?;
        ticket.issue_delay_wait_ticks = 0;
        self.next_accept_tick = next_instruction_tick(ticket)?;
        if ticket.uop_count != 0 {
            l0c.write_arbiter_mut()
                .enqueue(C220L0cMaster::Cube, ticket.accept_tick);
        }
        self.in_flight.push_back(C220CubeInFlight {
            instruction_id,
            next_uop_tick,
            ticket,
            pending_uops,
            l0c_port_granted: false,
            first_observation_tick: None,
            first_bubbles_remaining,
        });
        Ok(())
    }
}

fn next_instruction_tick(ticket: C220CubeTicket) -> Result<u64, C220CubeTimingError> {
    match ticket.last_uop_tick {
        Some(tick) => Ok(tick),
        None => ticket
            .accept_tick
            .checked_add(1)
            .ok_or(C220CubeTimingError::TimeOverflow),
    }
}

fn delay_ticket_from_uop(
    ticket: &mut C220CubeTicket,
    uop_id: u64,
    delay: u64,
    reason: ResourceWait,
) -> Result<(), C220CubeTimingError> {
    shift_ticket_from_uop(ticket, uop_id, delay)?;
    ticket.resource_wait_ticks = ticket
        .resource_wait_ticks
        .checked_add(delay)
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let counter = match reason {
        ResourceWait::InstructionOrder => &mut ticket.resource_waits.instruction_order,
        ResourceWait::HardwareFlag => &mut ticket.resource_waits.hardware_flag,
        ResourceWait::L0cWritePort => &mut ticket.resource_waits.l0c_write_port,
        ResourceWait::L0cUnitFlag => &mut ticket.resource_waits.l0c_unit_flag,
    };
    *counter = counter
        .checked_add(delay)
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    Ok(())
}

fn shift_ticket_from_uop(
    ticket: &mut C220CubeTicket,
    uop_id: u64,
    delay: u64,
) -> Result<(), C220CubeTimingError> {
    if uop_id == 0
        && let Some(tick) = ticket.first_uop_tick
    {
        ticket.first_uop_tick = Some(
            tick.checked_add(delay)
                .ok_or(C220CubeTimingError::TimeOverflow)?,
        );
    }
    if let Some(tick) = ticket.last_uop_tick {
        ticket.last_uop_tick = Some(
            tick.checked_add(delay)
                .ok_or(C220CubeTimingError::TimeOverflow)?,
        );
    }
    ticket.retire_tick = ticket
        .retire_tick
        .checked_add(delay)
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    Ok(())
}

fn validate_config(config: C220CubeConfig) -> Result<(), C220CubeTimingError> {
    if config.array_edge != C220_CUBE_ARRAY_EDGE {
        return Err(C220CubeTimingError::InvalidArrayEdge(config.array_edge));
    }
    if config.cube_spec_npe == 0 || 256 % config.cube_spec_npe != 0 {
        return Err(C220CubeTimingError::InvalidNpe(config.cube_spec_npe));
    }
    if config.fsm_version == C220CubeFsmVersion::V1 && config.cube_spec_npe != 256 {
        return Err(C220CubeTimingError::InvalidV1Npe(config.cube_spec_npe));
    }
    if config.cube_stage_num == 0 {
        return Err(C220CubeTimingError::InvalidStageCount);
    }
    Ok(())
}

fn schedule(
    config: C220CubeConfig,
    accept_tick: u64,
    instruction: C220CubeInstruction,
    parameters: C220MmadParameters,
    control: C220CubeTimingControl,
    previous_last_uop_tick: u64,
) -> Result<C220CubeTicket, C220CubeTimingError> {
    let geometry = instruction.geometry(parameters);
    let v1_frame_order = if config.v1_n2_mode || (config.v1_m_priority && control.fsm_m_priority) {
        C220CubeV1FrameOrder::MThenN
    } else {
        C220CubeV1FrameOrder::NThenM
    };
    let v1_f32_bubble = instruction.data_type == C220CubeDataType::F32F32
        && control.f32_mode == C220F32MmadMode::Fp32;
    let v1_dtype_bubbles_per_uop = u8::from(v1_f32_bubble)
        + u8::from(v1_f32_bubble && geometry.m_tiles == 1 && geometry.n_tiles == 1);
    let scale = 256 / u64::from(config.cube_spec_npe);
    let uop_count = geometry
        .native_uop_count
        .checked_mul(scale)
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    if uop_count == 0 {
        return Ok(C220CubeTicket {
            accept_tick,
            first_uop_tick: None,
            last_uop_tick: None,
            retire_tick: accept_tick,
            geometry,
            uop_count,
            fsm_bubbles: 0,
            sparse_bubbles: 0,
            resource_wait_ticks: 0,
            resource_waits: C220CubeResourceWaits::default(),
            issue_delay_wait_ticks: 0,
            issue_delay: control.issue_delay,
            fsm_version: config.fsm_version,
            v1_n2_mode: config.v1_n2_mode,
            v1_frame_order,
            v1_dtype_bubbles_per_uop,
            bias_checkpoint_uop: None,
            unit_flag_sync: parameters.xt_bits_55_56 >= 2,
        });
    }
    let v1_shape_bubbles = if geometry.m_tiles == 1 && geometry.n_tiles == 1 {
        uop_count
    } else if config.v1_n2_mode
        && !geometry.m_tiles.is_multiple_of(2)
        && !geometry.n_tiles.is_multiple_of(2)
    {
        u64::from(geometry.k_tiles)
    } else {
        0
    };
    let fsm_bubbles = match config.fsm_version {
        C220CubeFsmVersion::V0
            if !geometry.m_tiles.is_multiple_of(2) || !geometry.n_tiles.is_multiple_of(2) =>
        {
            u64::from(geometry.k_tiles.saturating_sub(1))
        }
        C220CubeFsmVersion::V1 => uop_count
            .checked_mul(u64::from(v1_dtype_bubbles_per_uop))
            .and_then(|bubbles| bubbles.checked_add(v1_shape_bubbles))
            .ok_or(C220CubeTimingError::TimeOverflow)?,
        C220CubeFsmVersion::V0 => 0,
    };
    let first_fsm_bubbles = match config.fsm_version {
        C220CubeFsmVersion::V0 => 0,
        C220CubeFsmVersion::V1 => {
            u64::from(v1_dtype_bubbles_per_uop)
                + u64::from(geometry.m_tiles == 1 && geometry.n_tiles == 1)
        }
    };
    let sparse_bubbles = u64::from(instruction.operation == C220CubeOperation::SparseMmad);
    let first_observation_tick = accept_tick
        .checked_add(1)
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let previous_issue_guard_tick = previous_last_uop_tick
        .checked_add(u64::from(control.issue_delay.previous_issue_guard_ticks))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let first_attempt_tick = first_observation_tick
        .checked_add(sparse_bubbles)
        .and_then(|tick| tick.checked_add(first_fsm_bubbles))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let delay_start_tick = first_observation_tick.max(previous_issue_guard_tick);
    let issue_delay_wait_ticks =
        if control.issue_delay.enabled && delay_start_tick <= first_attempt_tick {
            first_observation_tick
                .checked_add(u64::from(control.issue_delay.first_observation_delay_ticks))
                .ok_or(C220CubeTimingError::TimeOverflow)?
                .saturating_sub(delay_start_tick)
        } else {
            0
        };
    let first_uop_tick = accept_tick
        .checked_add(1)
        .and_then(|tick| tick.checked_add(issue_delay_wait_ticks))
        .and_then(|tick| tick.checked_add(sparse_bubbles))
        .and_then(|tick| tick.checked_add(first_fsm_bubbles))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let last_uop_tick = accept_tick
        .checked_add(uop_count)
        .and_then(|tick| tick.checked_add(issue_delay_wait_ticks))
        .and_then(|tick| tick.checked_add(fsm_bubbles))
        .and_then(|tick| tick.checked_add(sparse_bubbles))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let retire_tick = last_uop_tick
        .checked_add(u64::from(config.cube_stage_num - 1))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    Ok(C220CubeTicket {
        accept_tick,
        first_uop_tick: Some(first_uop_tick),
        last_uop_tick: Some(last_uop_tick),
        retire_tick,
        geometry,
        uop_count,
        fsm_bubbles,
        sparse_bubbles,
        resource_wait_ticks: 0,
        resource_waits: C220CubeResourceWaits::default(),
        issue_delay_wait_ticks,
        issue_delay: control.issue_delay,
        fsm_version: config.fsm_version,
        v1_n2_mode: config.v1_n2_mode,
        v1_frame_order,
        v1_dtype_bubbles_per_uop,
        bias_checkpoint_uop: parameters.xt_bit_62.then(|| {
            let last = (uop_count - 1) as u32;
            u64::from(if last == 0 {
                0
            } else {
                last.wrapping_sub(u32::from(geometry.k_tiles))
            })
        }),
        unit_flag_sync: parameters.xt_bits_55_56 >= 2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::cube::{C220CubeDataType, C220CubeOperation};

    const fn timing_control(spr3: u64) -> C220CubeTimingControl {
        C220CubeTimingControl::from_sprs(spr3, 0, 0)
    }

    #[test]
    fn empty_mmad_retires_without_waiting_for_an_older_pipeline_tail() {
        use crate::isa::c220::cube::C220CubeRegisterValues;
        use crate::sim::c220::cube::C220CubeV1UopPlanner;

        let instruction = C220CubeInstruction::decode((7 << 29) | (3 << 22)).unwrap();
        let parameters = instruction.parameters(C220CubeRegisterValues {
            xd: 0,
            xn: 0,
            xm: 0,
            xt: 16 | (16 << 12) | (16 << 24),
        });
        let mut pipeline = C220CubePipeline::new(C220CubeConfig::default()).unwrap();
        let mut l0c = C220L0c::new(1 << 20, 12).unwrap();
        let mut flags = C220HardwareFlagState::default();
        let older = pipeline
            .preview_issue(0, instruction, parameters, timing_control(0))
            .unwrap();
        pipeline
            .issue(
                older,
                C220CubeV1UopPlanner::new(older, instruction, parameters),
                0,
                &mut l0c,
            )
            .unwrap();
        let accept = pipeline.next_accept_tick();
        let empty_parameters = C220MmadParameters { m: 0, ..parameters };
        let empty = pipeline
            .preview_issue(accept, instruction, empty_parameters, timing_control(0))
            .unwrap();
        pipeline
            .issue(
                empty,
                C220CubeV1UopPlanner::new(empty, instruction, empty_parameters),
                1,
                &mut l0c,
            )
            .unwrap();
        assert!(accept < older.retire_tick);
        pipeline.advance_to(accept, &mut l0c, &mut flags).unwrap();
        assert_eq!(pipeline.last_retirements(), &[empty]);
        assert_eq!(pipeline.pending_retirement_count(), 1);
        pipeline
            .advance_to(older.retire_tick, &mut l0c, &mut flags)
            .unwrap();
        assert_eq!(pipeline.last_retirements(), &[older]);
        assert_eq!(pipeline.pending_retirement_count(), 0);
    }

    #[test]
    fn delayed_instructions_keep_uop_and_retirement_order_across_large_advances() {
        use crate::isa::c220::cube::C220CubeRegisterValues;
        use crate::isa::c220::hflag::C220HardwareFlagInstruction;
        let instruction = C220CubeInstruction::decode((7 << 29) | (3 << 22)).unwrap();
        let parameters = instruction.parameters(C220CubeRegisterValues {
            xd: 0,
            xn: 0,
            xm: 0,
            xt: 16 | (32 << 12) | (16 << 24),
        });
        let mut bulk = C220CubePipeline::new(C220CubeConfig::default()).unwrap();
        let mut memory = C220L0c::new(1 << 20, 12).unwrap();
        for id in 1..=2 {
            let control = if id == 1 {
                C220CubeTimingControl::from_sprs(0, (1 << 4) | (3 << 5), 0)
            } else {
                timing_control(0)
            };
            let ticket = bulk
                .preview_issue(bulk.next_accept_tick(), instruction, parameters, control)
                .unwrap();
            bulk.issue(
                ticket,
                crate::sim::c220::cube::C220CubeV1UopPlanner::new(ticket, instruction, parameters),
                id,
                &mut memory,
            )
            .unwrap();
        }
        let flag = (2 << 29) | (15 << 21) | (1 << 15) | (3 << 10) | (2 << 7);
        let set = C220HardwareFlagInstruction::decode(flag)
            .unwrap()
            .resolve(0, &[0; 32])
            .unwrap();
        let wait = C220HardwareFlagInstruction::decode(flag | (1 << 5))
            .unwrap()
            .resolve(0, &[0; 32])
            .unwrap();
        let mut flags = C220HardwareFlagState::default();
        flags.schedule_set(set, 20).unwrap();
        flags.enqueue_cube_wait(0, wait).unwrap();
        let (mut incremental, mut incremental_memory, mut incremental_flags) =
            (bulk.clone(), memory.clone(), flags.clone());
        let mut releases = Vec::new();
        let mut retirements = Vec::new();
        for tick in 0..=64 {
            incremental
                .advance_to(tick, &mut incremental_memory, &mut incremental_flags)
                .unwrap();
            releases.extend_from_slice(incremental.last_uop_releases());
            retirements.extend_from_slice(incremental.last_retirements());
        }
        bulk.advance_to(64, &mut memory, &mut flags).unwrap();
        assert_eq!(bulk.last_uop_releases(), releases);
        assert_eq!(bulk.last_retirements(), retirements);
        assert_eq!(memory, incremental_memory);
        assert_eq!(flags, incremental_flags);
        assert_eq!(
            releases
                .iter()
                .map(|release| release.instruction_id)
                .collect::<Vec<_>>(),
            [1, 1, 2, 2]
        );
        assert_eq!(releases[0].issue_tick, 25);
        assert_eq!(retirements[0].issue_delay_wait_ticks, 3);
        assert_eq!(retirements[0].resource_waits.hardware_flag, 20);
        for ticket in &retirements {
            let waits = ticket.resource_waits;
            assert_eq!(
                ticket.resource_wait_ticks,
                waits.instruction_order
                    + waits.hardware_flag
                    + waits.l0c_write_port
                    + waits.l0c_unit_flag
            );
        }
        assert!(
            releases
                .windows(2)
                .all(|pair| pair[0].issue_tick < pair[1].issue_tick)
        );
        assert_eq!(retirements.len(), 2);
        assert_eq!(bulk.pending_retirement_count(), 0);
    }

    #[test]
    fn first_uop_delay_guard_is_rechecked_during_bubbles() {
        for sparse in [false, true] {
            let instruction = C220CubeInstruction {
                word: 0,
                operation: if sparse {
                    C220CubeOperation::SparseMmad
                } else {
                    C220CubeOperation::Mmad
                },
                data_type: C220CubeDataType::F32F32,
                raw_data_type: 10,
                xd: 0,
                xn: 1,
                xm: 2,
                xt: 3,
            };
            let parameters =
                instruction.parameters(crate::isa::c220::cube::C220CubeRegisterValues {
                    xd: 0,
                    xn: 0,
                    xm: 0,
                    xt: 16 | (8 << 12) | (16 << 24),
                });
            for guard in [0, 11, 12, 14, 15, 16, 63] {
                let mut control = timing_control(0);
                control.issue_delay = C220CubeIssueDelay {
                    enabled: true,
                    previous_issue_guard_ticks: guard,
                    first_observation_delay_ticks: 10,
                };
                let ticket = schedule(
                    C220CubeConfig::default(),
                    10,
                    instruction,
                    parameters,
                    control,
                    0,
                )
                .unwrap();
                let mut tick = 11;
                let mut bubbles = 3 + u64::from(sparse);
                let mut wait = 0;
                loop {
                    if tick >= u64::from(guard) && tick < 21 {
                        wait += 1;
                    } else if bubbles != 0 {
                        bubbles -= 1;
                    } else {
                        break;
                    }
                    tick += 1;
                }
                assert_eq!(
                    ticket.first_uop_tick,
                    Some(tick),
                    "guard={guard} sparse={sparse}"
                );
                assert_eq!(ticket.issue_delay_wait_ticks, wait);
            }
        }
    }

    #[test]
    fn v0_mmad_keeps_the_pipeline_tail_out_of_the_accept_gap() {
        let instruction = C220CubeInstruction {
            word: 0,
            operation: C220CubeOperation::Mmad,
            data_type: C220CubeDataType::F16F32,
            raw_data_type: 3,
            xd: 0,
            xn: 1,
            xm: 2,
            xt: 3,
        };
        let parameters = instruction.parameters(crate::isa::c220::cube::C220CubeRegisterValues {
            xd: 0,
            xn: 0,
            xm: 0,
            xt: 16 | (32 << 12) | (16 << 24),
        });
        let mut pipeline = C220CubePipeline::new(C220CubeConfig {
            fsm_version: C220CubeFsmVersion::V0,
            ..C220CubeConfig::default()
        })
        .unwrap();
        let ticket = pipeline
            .preview_issue(10, instruction, parameters, timing_control(0))
            .unwrap();
        assert_eq!(ticket.uop_count, 2);
        assert_eq!(ticket.first_uop_tick, Some(11));
        assert_eq!(ticket.last_uop_tick, Some(13));
        assert_eq!(ticket.retire_tick, 34);
        let uops =
            crate::sim::c220::cube::C220CubeV0UopPlanner::new(ticket, instruction, parameters);
        let mut l0c = C220L0c::new(1 << 20, 12).unwrap();
        pipeline.issue(ticket, uops, 0, &mut l0c).unwrap();
        assert_eq!(pipeline.next_accept_tick(), 13);
        assert_eq!(pipeline.pending_drain_tick(), Some(34));
        let mut flags = C220HardwareFlagState::default();
        pipeline.advance_to(13, &mut l0c, &mut flags).unwrap();
        let next = pipeline
            .preview_issue(13, instruction, parameters, timing_control(0))
            .unwrap();
        assert_eq!(next.first_uop_tick, Some(14));
        pipeline
            .issue(
                next,
                crate::sim::c220::cube::C220CubeV0UopPlanner::new(next, instruction, parameters),
                1,
                &mut l0c,
            )
            .unwrap();
        assert_eq!(pipeline.pending_retirement_count(), 2);
    }

    #[test]
    fn v1_fp32_stalls_are_counted_before_each_uop() {
        let instruction = C220CubeInstruction {
            word: 0,
            operation: C220CubeOperation::Mmad,
            data_type: C220CubeDataType::F32F32,
            raw_data_type: 10,
            xd: 0,
            xn: 1,
            xm: 2,
            xt: 3,
        };
        let parameters = instruction.parameters(crate::isa::c220::cube::C220CubeRegisterValues {
            xd: 0,
            xn: 0,
            xm: 0,
            xt: 16 | (8 << 12) | (16 << 24),
        });
        let pipeline = C220CubePipeline::new(C220CubeConfig::default()).unwrap();

        let fp32 = pipeline
            .preview_issue(10, instruction, parameters, timing_control(0))
            .unwrap();
        assert_eq!(fp32.fsm_bubbles, 3);
        assert_eq!(fp32.first_uop_tick, Some(14));
        assert_eq!(fp32.last_uop_tick, Some(14));

        let hf32 = pipeline
            .preview_issue(10, instruction, parameters, timing_control(1 << 46))
            .unwrap();
        assert_eq!(hf32.fsm_bubbles, 1);
        assert_eq!(hf32.first_uop_tick, Some(12));
        assert_eq!(hf32.last_uop_tick, Some(12));

        let delayed = pipeline
            .preview_issue(
                10,
                instruction,
                parameters,
                C220CubeTimingControl::from_sprs(0, (1 << 4) | (3 << 1) | (4 << 5), 2 << 24),
            )
            .unwrap();
        assert_eq!(delayed.issue_delay_wait_ticks, 39);
        assert_eq!(delayed.first_uop_tick, Some(53));

        let m_priority = C220CubePipeline::new(C220CubeConfig {
            v1_m_priority: true,
            ..C220CubeConfig::default()
        })
        .unwrap();
        let ordinary = m_priority
            .preview_issue(10, instruction, parameters, timing_control(0))
            .unwrap();
        assert_eq!(ordinary.v1_frame_order, C220CubeV1FrameOrder::NThenM);
        let selected = m_priority
            .preview_issue(10, instruction, parameters, timing_control(1 << 51))
            .unwrap();
        assert_eq!(selected.v1_frame_order, C220CubeV1FrameOrder::MThenN);

        let n2 = C220CubePipeline::new(C220CubeConfig {
            v1_n2_mode: true,
            ..C220CubeConfig::default()
        })
        .unwrap();
        let odd_instruction = C220CubeInstruction {
            data_type: C220CubeDataType::F16F32,
            raw_data_type: 3,
            ..instruction
        };
        let odd_parameters = C220MmadParameters {
            m: 48,
            raw_k: 32,
            effective_k: 32,
            n: 48,
            ..parameters
        };
        let odd = n2
            .preview_issue(10, odd_instruction, odd_parameters, timing_control(0))
            .unwrap();
        assert_eq!(odd.uop_count, 18);
        assert_eq!(odd.fsm_bubbles, 2);
        assert_eq!(odd.planned_uop_tick(15), Some(26));
        assert_eq!(odd.planned_uop_tick(16), Some(28));
        assert_eq!(odd.planned_uop_tick(17), Some(30));

        let mut unit_pipeline = C220CubePipeline::new(C220CubeConfig::default()).unwrap();
        let unit_parameters = C220MmadParameters {
            xt_bits_55_56: 3,
            ..parameters
        };
        let unit_ticket = unit_pipeline
            .preview_issue(20, instruction, unit_parameters, timing_control(0))
            .unwrap();
        let uops = crate::sim::c220::cube::C220CubeV1UopPlanner::new(
            unit_ticket,
            instruction,
            unit_parameters,
        );
        let mut l0c = C220L0c::new(1 << 20, 12).unwrap();
        unit_pipeline.issue(unit_ticket, uops, 0, &mut l0c).unwrap();
        let mut hardware_flags = C220HardwareFlagState::default();
        assert!(l0c.write_arbiter_mut().grant(C220L0cMaster::Mte));
        unit_pipeline
            .advance_to(
                unit_ticket.first_uop_tick.unwrap(),
                &mut l0c,
                &mut hardware_flags,
            )
            .unwrap();
        assert!(unit_pipeline.last_uop_releases().is_empty());
        assert_eq!(
            unit_pipeline.last_l0c_stalls(),
            &[C220CubeL0cStall {
                instruction_id: 0,
                uop_id: 0,
                tick: unit_ticket.first_uop_tick.unwrap(),
                reason: C220CubeL0cStallReason::WritePort(C220L0cWritePortBlock::ActiveMaster {
                    master: C220L0cMaster::Mte,
                    outstanding: 1,
                }),
            }]
        );
        l0c.write_arbiter_mut().complete(C220L0cMaster::Mte);
        unit_pipeline
            .advance_to(
                unit_ticket.first_uop_tick.unwrap() + 1,
                &mut l0c,
                &mut hardware_flags,
            )
            .unwrap();
        assert!(unit_pipeline.last_l0c_stalls().is_empty());
        assert_eq!(unit_pipeline.last_uop_releases().len(), 1);
        assert_eq!(l0c.scoreboard().writer_flag_count(), 2);
        assert_eq!(l0c.read_banks().occupied_mask(), 0);
        assert_eq!(l0c.read_banks().future_masks()[1], 0x0001_0001);
        unit_pipeline
            .advance_to(
                unit_ticket.first_uop_tick.unwrap() + 2,
                &mut l0c,
                &mut hardware_flags,
            )
            .unwrap();
        assert_eq!(l0c.read_banks().occupied_mask(), 0x0001_0001);
    }
}
