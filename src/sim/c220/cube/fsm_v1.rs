use crate::isa::c220::cube::{C220CubeInstruction, C220MmadParameters};
use crate::sim::c220::cube::timing::C220CubeTicket;
use crate::sim::c220::cube::uop::{
    C220CubeL0cAccess, C220CubeL0cRequest, C220CubeTileIndices, C220CubeUop,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CubeV1UopPlanner {
    instruction: C220CubeInstruction,
    parameters: C220MmadParameters,
    uop_count: u64,
    pre_issue_bubbles: u8,
    next_uop_id: u64,
    m_tiles: u32,
    k_tiles: u32,
    n_tiles: u32,
    m_group_start: u32,
    n_group_start: u32,
    k_index: u32,
    group_offset: u32,
}

impl C220CubeV1UopPlanner {
    pub fn new(
        ticket: C220CubeTicket,
        instruction: C220CubeInstruction,
        parameters: C220MmadParameters,
    ) -> Self {
        let pre_issue_bubbles = ticket
            .fsm_bubbles
            .checked_div(ticket.uop_count)
            .and_then(|count| count.try_into().ok())
            .unwrap_or(0);
        Self {
            instruction,
            parameters,
            uop_count: ticket.uop_count,
            pre_issue_bubbles,
            next_uop_id: 0,
            m_tiles: u32::from(ticket.geometry.m_tiles),
            k_tiles: u32::from(ticket.geometry.k_tiles),
            n_tiles: u32::from(ticket.geometry.n_tiles),
            m_group_start: 0,
            n_group_start: 0,
            k_index: 0,
            group_offset: 0,
        }
    }

    fn group_size(remaining: u32) -> u32 {
        if remaining % 4 == 1 && remaining != 1 && remaining <= 5 {
            if remaining > 2 { 3 } else { 2 }
        } else {
            remaining.min(4)
        }
    }

    fn m_group_size(&self) -> u32 {
        if self.n_tiles == 1 {
            Self::group_size(self.m_tiles - self.m_group_start)
        } else {
            1
        }
    }

    fn n_group_size(&self) -> u32 {
        if self.n_tiles == 1 {
            1
        } else {
            Self::group_size(self.n_tiles - self.n_group_start)
        }
    }

    fn advance(&mut self, group_size: u32) {
        self.group_offset += 1;
        if self.group_offset < group_size {
            return;
        }
        self.group_offset = 0;
        self.k_index += 1;
        if self.k_index < self.k_tiles {
            return;
        }
        self.k_index = 0;
        self.m_group_start += self.m_group_size();
        if self.m_group_start < self.m_tiles {
            return;
        }
        self.m_group_start = 0;
        self.n_group_start += self.n_group_size();
    }

    fn l0c_request(&self, tile_index: u32, access: C220CubeL0cAccess) -> C220CubeL0cRequest {
        let bytes = self.instruction.data_type.l0c_request_bytes();
        C220CubeL0cRequest {
            address: (u64::from(tile_index) + u64::from(self.parameters.xd_low) / u64::from(bytes))
                * u64::from(bytes),
            bytes,
            access,
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
            || self.m_group_start >= self.m_tiles
            || self.n_group_start >= self.n_tiles
        {
            return None;
        }

        let id = self.next_uop_id;
        self.next_uop_id += 1;
        let varying_m = self.n_tiles == 1;
        let group_size = if varying_m {
            self.m_group_size()
        } else {
            self.n_group_size()
        };
        let m = self.m_group_start + if varying_m { self.group_offset } else { 0 };
        let n = self.n_group_start + if varying_m { 0 } else { self.group_offset };
        let k = self.k_index;
        let indices = C220CubeTileIndices {
            l0a: m * self.k_tiles + k,
            l0b: k * self.n_tiles + n,
            l0c: m * self.n_tiles + n,
        };
        let first_in_group = self.group_offset == 0;
        let reads_l0a = varying_m || first_in_group;
        let reads_l0b = !varying_m || first_in_group;
        let reads_l0c = k == 0 && !self.parameters.xt_bit_63;
        let writes_l0c = k + 1 == self.k_tiles;
        let uop = C220CubeUop {
            id,
            pre_issue_bubbles: self.pre_issue_bubbles,
            tile_indices: Some(indices),
            reads_l0a,
            reads_l0b,
            acquires_l0c_write_port: id == 0,
            l0c_read: reads_l0c.then(|| self.l0c_request(indices.l0c, C220CubeL0cAccess::Read)),
            l0c_write: writes_l0c.then(|| self.l0c_request(indices.l0c, C220CubeL0cAccess::Write)),
        };
        self.advance(group_size);
        Some(uop)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.uop_count.saturating_sub(self.next_uop_id);
        let remaining = usize::try_from(remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for C220CubeV1UopPlanner {}
