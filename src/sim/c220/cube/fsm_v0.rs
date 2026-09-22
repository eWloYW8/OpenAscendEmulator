use crate::isa::c220::cube::{C220CubeInstruction, C220MmadParameters};
use crate::sim::c220::cube::timing::C220CubeTicket;
use crate::sim::c220::cube::uop::{C220CubeL0cAccess, C220CubeL0cRequest, C220CubeUop};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220CubeV0UopPlanner {
    instruction: C220CubeInstruction,
    parameters: C220MmadParameters,
    uop_count: u64,
    next_uop_id: u64,
    original_m_tiles: u32,
    original_k_tiles: u32,
    original_n_tiles: u32,
    remaining_m_tiles: u32,
    remaining_k_tiles: u32,
    remaining_n_tiles: u32,
    traversal_mode: u32,
    compact_n: u32,
    fraction_cursor: u32,
    fraction_span: u32,
    l0c_tile_base: u32,
    prior_k_fold: bool,
}

impl C220CubeV0UopPlanner {
    pub fn new(
        ticket: C220CubeTicket,
        instruction: C220CubeInstruction,
        parameters: C220MmadParameters,
    ) -> Self {
        let m_tiles = u32::from(ticket.geometry.m_tiles);
        let k_tiles = u32::from(ticket.geometry.k_tiles);
        let n_tiles = u32::from(ticket.geometry.n_tiles);
        Self {
            instruction,
            parameters,
            uop_count: ticket.uop_count,
            next_uop_id: 0,
            original_m_tiles: m_tiles,
            original_k_tiles: k_tiles,
            original_n_tiles: n_tiles,
            remaining_m_tiles: m_tiles,
            remaining_k_tiles: k_tiles,
            remaining_n_tiles: n_tiles,
            traversal_mode: 1,
            compact_n: u32::from(n_tiles <= 2),
            fraction_cursor: 0,
            fraction_span: 0,
            l0c_tile_base: 0,
            prior_k_fold: false,
        }
    }

    fn next_l0c_read_index(&mut self) -> Option<u32> {
        if self.original_m_tiles == 0 || self.original_k_tiles == 0 || self.original_n_tiles == 0 {
            return None;
        }

        let m_shape = if self.remaining_m_tiles == 1 { 1 } else { 2 };
        let (mut span, k_shape) = if self.remaining_k_tiles == 1 {
            (m_shape, 1)
        } else {
            (2 * m_shape, 2)
        };
        let n_shape = if self.original_n_tiles == 1 {
            1
        } else {
            span *= 2;
            2
        };
        self.fraction_span = span;

        if self.remaining_k_tiles == self.original_k_tiles
            && self.fraction_cursor >= self.fraction_span
        {
            self.l0c_tile_base = self.l0c_tile_base.wrapping_add(
                if self.original_m_tiles <= 1 || self.original_n_tiles > 2 {
                    if self.original_n_tiles <= 2 { 1 } else { 2 }
                } else {
                    2 * self.original_n_tiles
                },
            );
        }

        match self.traversal_mode {
            0 => self.jump_b_column(),
            1 => self.jump_a_column(),
            _ => {}
        }
        if self.fraction_cursor >= self.fraction_span {
            self.traversal_mode = self.compact_n;
            self.fraction_cursor = 0;
        }
        self.prior_k_fold = self.remaining_k_tiles <= 2;

        let (mapped, basic_valid) = self.basic_l0c_map(m_shape, k_shape, n_shape);
        let index = mapped.wrapping_add(self.l0c_tile_base);
        let valid = self.remaining_k_tiles == self.original_k_tiles
            && self.original_m_tiles.wrapping_mul(self.original_n_tiles) > index
            && basic_valid;
        self.fraction_cursor = self.fraction_cursor.wrapping_add(1);
        valid.then_some(index)
    }

    fn jump_a_column(&mut self) {
        if self.fraction_cursor < self.fraction_span {
            return;
        }
        if self.prior_k_fold {
            if self.remaining_n_tiles <= 2 {
                self.remaining_n_tiles = self.original_n_tiles;
                self.remaining_k_tiles = self.original_k_tiles;
                self.remaining_m_tiles = self.remaining_m_tiles.wrapping_sub(2);
                self.traversal_mode = 1;
                return;
            }
        } else if self.remaining_n_tiles <= 2 {
            self.remaining_k_tiles = self.remaining_k_tiles.wrapping_sub(2);
            self.traversal_mode = 1;
            return;
        }
        self.traversal_mode = 0;
        self.remaining_n_tiles = self.remaining_n_tiles.wrapping_sub(2);
    }

    fn jump_b_column(&mut self) {
        if self.fraction_cursor < self.fraction_span {
            return;
        }
        if self.prior_k_fold {
            self.remaining_k_tiles = self.original_k_tiles;
            if self.remaining_n_tiles > 2 {
                self.remaining_n_tiles = self.remaining_n_tiles.wrapping_sub(2);
            } else {
                self.remaining_n_tiles = self.original_n_tiles;
                self.remaining_m_tiles = self.remaining_m_tiles.wrapping_sub(2);
            }
        } else {
            self.remaining_n_tiles = self.remaining_n_tiles.wrapping_add(2);
            self.remaining_k_tiles = self.remaining_k_tiles.wrapping_sub(2);
        }
        self.traversal_mode = 1;
    }

    fn basic_l0c_map(&self, m_shape: u32, k_shape: u32, n_shape: u32) -> (u32, bool) {
        let cursor = self.fraction_cursor;
        match self.fraction_span {
            1 => (0, true),
            2 if k_shape == 2 => (0, cursor == 0),
            2 => (cursor, true),
            4 if m_shape == 2 && k_shape == 2 => match cursor {
                0 => (0, true),
                1 => (1, true),
                _ => (0, false),
            },
            4 if k_shape == 2 && n_shape == 2 => (cursor, cursor <= 1),
            4 if m_shape == 2 && n_shape == 2 => match cursor {
                0 => (0, true),
                1 => (self.original_n_tiles, true),
                2 => (1, true),
                3 => (self.original_n_tiles.wrapping_add(1), true),
                _ => (0, false),
            },
            8 => match cursor {
                0 => (0, true),
                1 => (self.original_n_tiles, true),
                2 => (1, true),
                3 => (self.original_n_tiles.wrapping_add(1), true),
                _ => (0, false),
            },
            _ => (0, false),
        }
    }
}

impl Iterator for C220CubeV0UopPlanner {
    type Item = C220CubeUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_uop_id >= self.uop_count {
            return None;
        }
        let id = self.next_uop_id;
        self.next_uop_id += 1;
        let read_index = self.next_l0c_read_index();
        let request_bytes = self.instruction.data_type.l0c_request_bytes();
        let l0c_read = read_index
            .filter(|_| !self.parameters.xt_bit_63)
            .map(|index| C220CubeL0cRequest {
                address: (u64::from(index)
                    + u64::from(self.parameters.xd_low) / u64::from(request_bytes))
                    * u64::from(request_bytes),
                bytes: request_bytes,
                access: C220CubeL0cAccess::Read,
            });
        let bubble_start = self
            .uop_count
            .saturating_sub(u64::from(self.original_k_tiles));
        Some(C220CubeUop {
            id,
            pre_issue_bubbles: u8::from(
                (self.original_m_tiles | self.original_n_tiles) & 1 != 0 && id > bubble_start,
            ),
            tile_indices: None,
            reads_l0a: true,
            reads_l0b: true,
            acquires_l0c_write_port: id == 0,
            l0c_read,
            l0c_write: None,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.uop_count.saturating_sub(self.next_uop_id);
        let remaining = usize::try_from(remaining).unwrap_or(usize::MAX);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for C220CubeV0UopPlanner {}
