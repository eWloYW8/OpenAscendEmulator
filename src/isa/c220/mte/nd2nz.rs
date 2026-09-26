#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Nd2NzFormat {
    B8,
    B16,
    B32,
    B32Wide,
}

impl C220Nd2NzFormat {
    pub const fn element_bytes(self) -> u32 {
        match self {
            Self::B8 => 1,
            Self::B16 => 2,
            Self::B32 | Self::B32Wide => 4,
        }
    }

    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::B32Wide => 64,
            _ => 32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzInstruction {
    pub word: u32,
    pub format: C220Nd2NzFormat,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
    pub stride_register: u8,
}

impl C220Nd2NzInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 1 || (word >> 22) & 31 != 12 {
            return None;
        }
        Some(Self {
            word,
            format: match word & 3 {
                0 => C220Nd2NzFormat::B8,
                1 => C220Nd2NzFormat::B16,
                2 => C220Nd2NzFormat::B32,
                _ => C220Nd2NzFormat::B32Wide,
            },
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
            stride_register: ((word >> 2) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220Nd2NzTransfer {
        C220Nd2NzTransfer {
            instruction: self,
            source_base: registers[usize::from(self.source_register)],
            destination_base: registers[usize::from(self.destination_register)],
            xm: registers[usize::from(self.descriptor_register)],
            xt: registers[usize::from(self.stride_register)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzTransfer {
    pub instruction: C220Nd2NzInstruction,
    pub source_base: u64,
    pub destination_base: u64,
    pub xm: u64,
    pub xt: u64,
}

impl C220Nd2NzTransfer {
    pub const fn sid(self) -> u8 {
        (self.xm & 15) as u8
    }
    pub const fn matrix_count(self) -> u16 {
        ((self.xm >> 4) & 4095) as u16
    }
    pub const fn rows(self) -> u16 {
        (self.xm >> 16) as u16
    }
    pub const fn columns(self) -> u16 {
        (self.xm >> 32) as u16
    }
    pub const fn source_matrix_stride(self) -> u16 {
        (self.xm >> 48) as u16
    }
    pub const fn source_row_stride(self) -> u16 {
        self.xt as u16
    }
    pub const fn destination_column_stride(self) -> u16 {
        (self.xt >> 16) as u16
    }
    pub const fn destination_row_stride(self) -> u16 {
        (self.xt >> 32) as u16
    }
    pub const fn destination_matrix_stride(self) -> u16 {
        (self.xt >> 48) as u16
    }
    pub const fn row_bytes(self) -> u32 {
        self.columns() as u32 * self.instruction.format.element_bytes()
    }
    pub const fn blocks_per_row(self) -> u32 {
        self.row_bytes()
            .div_ceil(self.instruction.format.block_bytes())
    }
    pub const fn segment_count(self) -> u64 {
        self.matrix_count() as u64 * self.rows() as u64 * self.blocks_per_row() as u64
    }
    pub const fn is_disabled(self) -> bool {
        self.segment_count() == 0
    }

    pub fn segment(self, index: u64) -> Option<C220Nd2NzSegment> {
        if index >= self.segment_count() {
            return None;
        }
        let blocks = u64::from(self.blocks_per_row());
        let block = (index % blocks) as u32;
        let row = ((index / blocks) % u64::from(self.rows())) as u16;
        let matrix = (index / blocks / u64::from(self.rows())) as u16;
        let element_bytes = self.instruction.format.element_bytes();
        let block_bytes = self.instruction.format.block_bytes();
        let source_matrix = u32::from(matrix)
            .wrapping_mul(u32::from(self.source_matrix_stride()))
            .wrapping_mul(element_bytes);
        let destination_matrix = u32::from(matrix)
            .wrapping_mul(u32::from(self.destination_matrix_stride()))
            .wrapping_mul(element_bytes);
        let source_row = u32::from(row)
            .wrapping_mul(u32::from(self.source_row_stride()))
            .wrapping_mul(element_bytes);
        let destination_row = u32::from(row)
            .wrapping_mul(u32::from(self.destination_row_stride()))
            .wrapping_mul(block_bytes);
        let (column_offset, half_offset) = if block_bytes == 32 {
            (
                block
                    .wrapping_mul(u32::from(self.destination_column_stride()))
                    .wrapping_mul(32),
                0,
            )
        } else {
            (
                (block / 2)
                    .wrapping_mul(u32::from(self.destination_column_stride()))
                    .wrapping_mul(64),
                (block & 1) * 32,
            )
        };
        Some(C220Nd2NzSegment {
            matrix_index: matrix,
            row_index: row,
            block_index: block,
            source_address: self
                .source_base
                .wrapping_add(u64::from(source_matrix))
                .wrapping_add(u64::from(source_row))
                .wrapping_add(u64::from(block * 32)),
            destination_address: self
                .destination_base
                .wrapping_add(u64::from(destination_matrix))
                .wrapping_add(u64::from(destination_row))
                .wrapping_add(u64::from(column_offset))
                .wrapping_add(u64::from(half_offset)),
            input_bytes: (self.row_bytes() - block * block_bytes).min(block_bytes),
            output_bytes: block_bytes,
        })
    }

    pub fn segments(self) -> impl Iterator<Item = C220Nd2NzSegment> {
        (0..self.segment_count()).map(move |index| self.segment(index).expect("bounded segment"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Nd2NzSegment {
    pub matrix_index: u16,
    pub row_index: u16,
    pub block_index: u32,
    pub source_address: u64,
    pub destination_address: u64,
    pub input_bytes: u32,
    pub output_bytes: u32,
}
