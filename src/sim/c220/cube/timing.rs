use std::collections::VecDeque;

use thiserror::Error;

use crate::isa::c220::cube::{
    C220_CUBE_ARRAY_EDGE, C220CubeDataType, C220CubeGeometry, C220CubeInstruction,
    C220CubeOperation, C220MmadParameters,
};
use crate::sim::c220::cube::execute::{C220CubeControl, C220F32MmadMode};
use crate::sim::c220::memory::{C220L0cMaster, C220L0cWriteArbiter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeFsmVersion {
    V0,
    V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeConfig {
    pub array_edge: u16,
    pub cube_spec_npe: u16,
    pub cube_stage_num: u16,
    pub fsm_version: C220CubeFsmVersion,
}

impl Default for C220CubeConfig {
    fn default() -> Self {
        Self {
            array_edge: C220_CUBE_ARRAY_EDGE,
            cube_spec_npe: 256,
            cube_stage_num: 22,
            fsm_version: C220CubeFsmVersion::V1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeTicket {
    pub accept_tick: u64,
    pub first_uop_tick: Option<u64>,
    pub last_uop_tick: Option<u64>,
    pub retire_tick: u64,
    pub geometry: C220CubeGeometry,
    pub uop_count: u64,
    pub fsm_bubbles: u64,
    pub sparse_bubbles: u64,
    pub resource_wait_ticks: u64,
    pub fsm_version: C220CubeFsmVersion,
}

impl C220CubeTicket {
    pub fn uop_issue_tick(self, uop_id: u64) -> Option<u64> {
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
            C220CubeFsmVersion::V1 => self
                .fsm_bubbles
                .checked_div(self.uop_count)?
                .checked_mul(uop_id.checked_add(1)?)?,
        };
        self.accept_tick
            .checked_add(1)?
            .checked_add(self.sparse_bubbles)?
            .checked_add(self.resource_wait_ticks)?
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct C220CubeInFlight {
    ticket: C220CubeTicket,
    l0c_port_granted: bool,
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
        })
    }

    pub const fn config(&self) -> C220CubeConfig {
        self.config
    }

    pub const fn next_accept_tick(&self) -> u64 {
        self.next_accept_tick
    }

    pub fn pending_drain_tick(&self) -> Option<u64> {
        self.in_flight
            .back()
            .map(|flight| flight.ticket.retire_tick)
    }

    pub fn pending_retirement_count(&self) -> usize {
        self.in_flight.len()
    }

    pub fn last_retirements(&self) -> &[C220CubeTicket] {
        &self.last_retirements
    }

    pub fn advance_to(
        &mut self,
        tick: u64,
        l0c_arbiter: &mut C220L0cWriteArbiter,
    ) -> Result<&[C220CubeTicket], C220CubeTimingError> {
        if tick < self.now {
            return Err(C220CubeTimingError::TimeReversed {
                requested: tick,
                previous: self.now,
            });
        }
        self.now = tick;
        self.last_retirements.clear();

        for flight in &mut self.in_flight {
            let Some(first_uop_tick) = flight.ticket.first_uop_tick else {
                continue;
            };
            if flight.l0c_port_granted || first_uop_tick > tick {
                continue;
            }
            if l0c_arbiter.grant(C220L0cMaster::Cube) {
                flight.l0c_port_granted = true;
                continue;
            }
            let resume_tick = tick
                .checked_add(1)
                .ok_or(C220CubeTimingError::TimeOverflow)?;
            let delay = resume_tick
                .checked_sub(first_uop_tick)
                .ok_or(C220CubeTimingError::TimeOverflow)?;
            delay_ticket(&mut flight.ticket, delay)?;
            self.next_accept_tick = self
                .next_accept_tick
                .max(next_instruction_tick(flight.ticket)?);
        }

        while self.in_flight.front().is_some_and(|flight| {
            (flight.l0c_port_granted || flight.ticket.uop_count == 0)
                && flight.ticket.retire_tick <= tick
        }) {
            let flight = self
                .in_flight
                .pop_front()
                .expect("front Cube flight exists");
            if flight.l0c_port_granted {
                l0c_arbiter.complete(C220L0cMaster::Cube);
            }
            self.last_retirements.push(flight.ticket);
        }
        Ok(&self.last_retirements)
    }

    pub fn preview_issue(
        &self,
        accept_tick: u64,
        instruction: C220CubeInstruction,
        parameters: C220MmadParameters,
        control: C220CubeControl,
    ) -> Result<C220CubeTicket, C220CubeTimingError> {
        if accept_tick < self.next_accept_tick {
            return Err(C220CubeTimingError::Busy {
                ready_tick: self.next_accept_tick,
            });
        }
        schedule(self.config, accept_tick, instruction, parameters, control)
    }

    pub fn issue(
        &mut self,
        ticket: C220CubeTicket,
        l0c_arbiter: &mut C220L0cWriteArbiter,
    ) -> Result<(), C220CubeTimingError> {
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
        self.next_accept_tick = next_instruction_tick(ticket)?;
        if ticket.uop_count != 0 {
            l0c_arbiter.enqueue(C220L0cMaster::Cube, ticket.accept_tick);
        }
        self.in_flight.push_back(C220CubeInFlight {
            ticket,
            l0c_port_granted: false,
        });
        Ok(())
    }
}

fn next_instruction_tick(ticket: C220CubeTicket) -> Result<u64, C220CubeTimingError> {
    ticket
        .last_uop_tick
        .unwrap_or(ticket.accept_tick)
        .checked_add(1)
        .ok_or(C220CubeTimingError::TimeOverflow)
}

fn delay_ticket(ticket: &mut C220CubeTicket, delay: u64) -> Result<(), C220CubeTimingError> {
    if let Some(tick) = ticket.first_uop_tick {
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
    ticket.resource_wait_ticks = ticket
        .resource_wait_ticks
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
    control: C220CubeControl,
) -> Result<C220CubeTicket, C220CubeTimingError> {
    let geometry = instruction.geometry(parameters);
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
            fsm_version: config.fsm_version,
        });
    }
    let v1_shape_bubble = geometry.m_tiles == 1 && geometry.n_tiles == 1;
    let v1_f32_bubble = instruction.data_type == C220CubeDataType::F32F32
        && control.f32_mode == C220F32MmadMode::Fp32;
    let v1_bubbles_per_uop = u64::from(v1_shape_bubble)
        + u64::from(v1_f32_bubble)
        + u64::from(v1_shape_bubble && v1_f32_bubble);
    let fsm_bubbles = match config.fsm_version {
        C220CubeFsmVersion::V0
            if !geometry.m_tiles.is_multiple_of(2) || !geometry.n_tiles.is_multiple_of(2) =>
        {
            u64::from(geometry.k_tiles.saturating_sub(1))
        }
        C220CubeFsmVersion::V1 => uop_count
            .checked_mul(v1_bubbles_per_uop)
            .ok_or(C220CubeTimingError::TimeOverflow)?,
        C220CubeFsmVersion::V0 => 0,
    };
    let first_fsm_bubbles = match config.fsm_version {
        C220CubeFsmVersion::V0 => 0,
        C220CubeFsmVersion::V1 => v1_bubbles_per_uop,
    };
    let sparse_bubbles = u64::from(instruction.operation == C220CubeOperation::SparseMmad);
    let first_uop_tick = accept_tick
        .checked_add(1)
        .and_then(|tick| tick.checked_add(sparse_bubbles))
        .and_then(|tick| tick.checked_add(first_fsm_bubbles))
        .ok_or(C220CubeTimingError::TimeOverflow)?;
    let last_uop_tick = accept_tick
        .checked_add(uop_count)
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
        fsm_version: config.fsm_version,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::cube::{C220CubeDataType, C220CubeOperation};

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
            .preview_issue(10, instruction, parameters, C220CubeControl::from_spr3(0))
            .unwrap();
        assert_eq!(ticket.uop_count, 2);
        assert_eq!(ticket.first_uop_tick, Some(11));
        assert_eq!(ticket.last_uop_tick, Some(13));
        assert_eq!(ticket.retire_tick, 34);
        let mut arbiter = C220L0cWriteArbiter::default();
        pipeline.issue(ticket, &mut arbiter).unwrap();
        assert_eq!(pipeline.next_accept_tick(), 14);
        assert_eq!(pipeline.pending_drain_tick(), Some(34));
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
            .preview_issue(10, instruction, parameters, C220CubeControl::from_spr3(0))
            .unwrap();
        assert_eq!(fp32.fsm_bubbles, 3);
        assert_eq!(fp32.first_uop_tick, Some(14));
        assert_eq!(fp32.last_uop_tick, Some(14));

        let hf32 = pipeline
            .preview_issue(
                10,
                instruction,
                parameters,
                C220CubeControl::from_spr3(1 << 46),
            )
            .unwrap();
        assert_eq!(hf32.fsm_bubbles, 1);
        assert_eq!(hf32.first_uop_tick, Some(12));
        assert_eq!(hf32.last_uop_tick, Some(12));
    }
}
