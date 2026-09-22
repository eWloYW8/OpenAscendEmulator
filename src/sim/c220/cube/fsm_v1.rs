use crate::isa::c220::cube::{C220CubeInstruction, C220MmadParameters};
use crate::sim::c220::cube::C220CubeV1FrameOrder;
use crate::sim::c220::cube::timing::C220CubeTicket;
use crate::sim::c220::cube::uop::{
    C220CubeL0cAccess, C220CubeL0cRequest, C220CubeTileIndices, C220CubeUnitFlagMode, C220CubeUop,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CubeV1UopPlanner {
    instruction: C220CubeInstruction,
    parameters: C220MmadParameters,
    uop_count: u64,
    dtype_bubbles_per_uop: u8,
    next_uop_id: u64,
    m_tiles: u32,
    k_tiles: u32,
    n_tiles: u32,
    n2_mode: bool,
    frame_order: C220CubeV1FrameOrder,
    m_frame_start: u32,
    n_frame_start: u32,
    k_index: u32,
    frame_offset: u32,
}

impl C220CubeV1UopPlanner {
    pub fn new(
        ticket: C220CubeTicket,
        instruction: C220CubeInstruction,
        parameters: C220MmadParameters,
    ) -> Self {
        Self {
            instruction,
            parameters,
            uop_count: ticket.uop_count,
            dtype_bubbles_per_uop: ticket.v1_dtype_bubbles_per_uop,
            next_uop_id: 0,
            m_tiles: u32::from(ticket.geometry.m_tiles),
            k_tiles: u32::from(ticket.geometry.k_tiles),
            n_tiles: u32::from(ticket.geometry.n_tiles),
            n2_mode: ticket.v1_n2_mode,
            frame_order: ticket.v1_frame_order,
            m_frame_start: 0,
            n_frame_start: 0,
            k_index: 0,
            frame_offset: 0,
        }
    }

    fn group_size(remaining: u32) -> u32 {
        if remaining % 4 == 1 && remaining != 1 && remaining <= 5 {
            if remaining > 2 { 3 } else { 2 }
        } else {
            remaining.min(4)
        }
    }

    fn frame_shape(&self) -> (u32, u32) {
        let remaining_m = self.m_tiles - self.m_frame_start;
        let remaining_n = self.n_tiles - self.n_frame_start;
        if self.n2_mode {
            let m = if remaining_n == 1 {
                remaining_m.min(2)
            } else {
                1
            };
            return (m, remaining_n.min(2));
        }
        if self.n_tiles == 1 {
            (Self::group_size(remaining_m), 1)
        } else {
            (1, Self::group_size(remaining_n))
        }
    }

    fn advance(&mut self, frame_m_tiles: u32, frame_n_tiles: u32) {
        let frame_uops = frame_m_tiles * frame_n_tiles;
        self.frame_offset += 1;
        if self.frame_offset < frame_uops {
            return;
        }
        self.frame_offset = 0;
        self.k_index += 1;
        if self.k_index < self.k_tiles {
            return;
        }
        self.k_index = 0;
        match self.frame_order {
            C220CubeV1FrameOrder::NThenM => {
                self.n_frame_start += frame_n_tiles;
                if self.n_frame_start < self.n_tiles {
                    return;
                }
                self.n_frame_start = 0;
                self.m_frame_start += frame_m_tiles;
            }
            C220CubeV1FrameOrder::MThenN => {
                self.m_frame_start += frame_m_tiles;
                if self.m_frame_start < self.m_tiles {
                    return;
                }
                self.m_frame_start = 0;
                self.n_frame_start += frame_n_tiles;
            }
        }
    }

    fn l0c_request(&self, tile_index: u32, access: C220CubeL0cAccess) -> C220CubeL0cRequest {
        let bytes = self.instruction.data_type.l0c_request_bytes();
        C220CubeL0cRequest {
            address: (u64::from(tile_index) + u64::from(self.parameters.xd_low) / u64::from(bytes))
                * u64::from(bytes),
            bytes,
            access,
            unit_flags: if access == C220CubeL0cAccess::Write {
                C220CubeUnitFlagMode::from_xt_bits(self.parameters.xt_bits_55_56)
            } else {
                C220CubeUnitFlagMode::Disabled
            },
        }
    }
}

impl Iterator for C220CubeV1UopPlanner {
    type Item = C220CubeUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_uop_id >= self.uop_count
            || self.m_tiles == 0
            || self.k_tiles == 0
            || self.n_tiles == 0
            || self.m_frame_start >= self.m_tiles
            || self.n_frame_start >= self.n_tiles
        {
            return None;
        }

        let id = self.next_uop_id;
        self.next_uop_id += 1;
        let (frame_m_tiles, frame_n_tiles) = self.frame_shape();
        let varying_m = frame_m_tiles > 1;
        let m = self.m_frame_start + if varying_m { self.frame_offset } else { 0 };
        let n = self.n_frame_start + if varying_m { 0 } else { self.frame_offset };
        let k = self.k_index;
        let indices = C220CubeTileIndices {
            l0a: m * self.k_tiles + k,
            l0b: k * self.n_tiles + n,
            l0c: m * self.n_tiles + n,
        };
        let first_in_frame = self.frame_offset == 0;
        let reads_l0a = varying_m || first_in_frame;
        let reads_l0b = !varying_m || first_in_frame;
        let reads_l0c = k == 0 && !self.parameters.xt_bit_62 && !self.parameters.xt_bit_63;
        let writes_l0c = k + 1 == self.k_tiles;
        let shape_bubble = if self.n2_mode {
            !self.m_tiles.is_multiple_of(2)
                && !self.n_tiles.is_multiple_of(2)
                && indices.l0c == self.m_tiles * self.n_tiles - 1
        } else {
            self.m_tiles == 1 && self.n_tiles == 1
        };
        let uop = C220CubeUop {
            id,
            pre_issue_bubbles: self.dtype_bubbles_per_uop + u8::from(shape_bubble),
            tile_indices: Some(indices),
            reads_l0a,
            reads_l0b,
            acquires_l0c_write_port: id == 0,
            l0c_read: reads_l0c.then(|| self.l0c_request(indices.l0c, C220CubeL0cAccess::Read)),
            l0c_write: writes_l0c.then(|| self.l0c_request(indices.l0c, C220CubeL0cAccess::Write)),
        };
        self.advance(frame_m_tiles, frame_n_tiles);
        Some(uop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.uop_count.saturating_sub(self.next_uop_id);
        let remaining = usize::try_from(remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for C220CubeV1UopPlanner {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::cube::{C220CubeDataType, C220CubeGeometry, C220CubeOperation};

    fn l0c_indices(
        m_tiles: u16,
        k_tiles: u16,
        n_tiles: u16,
        n2_mode: bool,
        frame_order: C220CubeV1FrameOrder,
    ) -> Vec<u32> {
        let uop_count = u64::from(m_tiles) * u64::from(k_tiles) * u64::from(n_tiles);
        let ticket = C220CubeTicket {
            accept_tick: 0,
            first_uop_tick: Some(1),
            last_uop_tick: Some(uop_count),
            retire_tick: uop_count,
            geometry: C220CubeGeometry {
                m_tiles,
                k_tiles,
                n_tiles,
                k_tile_elements: 16,
                native_uop_count: uop_count,
            },
            uop_count,
            fsm_bubbles: 0,
            sparse_bubbles: 0,
            resource_wait_ticks: 0,
            issue_delay_wait_ticks: 0,
            issue_delay: crate::sim::c220::cube::C220CubeIssueDelay::default(),
            fsm_version: crate::sim::c220::cube::C220CubeFsmVersion::V1,
            v1_n2_mode: n2_mode,
            v1_frame_order: frame_order,
            v1_dtype_bubbles_per_uop: 0,
        };
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
        let parameters = C220MmadParameters {
            xd_low: 0,
            xd_high: 0,
            xn: 0,
            xm: 0,
            m: m_tiles * 16,
            raw_k: k_tiles * 16,
            effective_k: k_tiles * 16,
            n: n_tiles * 16,
            xt_bits_44_50: 0,
            xt_bits_55_56: 0,
            xt_bit_58: false,
            xt_bit_62: false,
            xt_bit_63: false,
        };
        C220CubeV1UopPlanner::new(ticket, instruction, parameters)
            .map(|uop| uop.tile_indices.expect("V1 uop has tile indices").l0c)
            .collect()
    }

    #[test]
    fn follows_configured_big_frame_order_and_n2_shapes() {
        assert_eq!(
            l0c_indices(2, 1, 5, false, C220CubeV1FrameOrder::NThenM),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        );
        assert_eq!(
            l0c_indices(2, 1, 5, false, C220CubeV1FrameOrder::MThenN),
            [0, 1, 2, 5, 6, 7, 3, 4, 8, 9]
        );
        assert_eq!(
            l0c_indices(3, 1, 3, true, C220CubeV1FrameOrder::MThenN),
            [0, 1, 3, 4, 6, 7, 2, 5, 8]
        );
    }
}
