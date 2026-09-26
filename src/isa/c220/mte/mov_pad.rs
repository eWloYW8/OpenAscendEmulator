use super::C220MovDirection;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovPadInstruction {
    pub word: u32,
    pub direction: C220MovDirection,
    pub element_bytes: u8,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
    pub stride_register: u8,
}

impl C220MovPadInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 1 || (word >> 23) & 15 != 7 {
            return None;
        }
        let element_bytes = match word & 3 {
            0 => 1,
            1 => 2,
            2 => 4,
            _ => return None,
        };
        Some(Self {
            word,
            direction: if word & (1 << 22) == 0 {
                C220MovDirection::HbmToUb
            } else {
                C220MovDirection::UbToHbm
            },
            element_bytes,
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
            stride_register: ((word >> 2) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220MovPadTransfer {
        C220MovPadTransfer {
            instruction: self,
            source_base: registers[usize::from(self.source_register)],
            destination_base: registers[usize::from(self.destination_register)],
            xm: registers[usize::from(self.descriptor_register)],
            xt: registers[usize::from(self.stride_register)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovPadTransfer {
    pub instruction: C220MovPadInstruction,
    pub source_base: u64,
    pub destination_base: u64,
    pub xm: u64,
    pub xt: u64,
}

impl C220MovPadTransfer {
    pub const fn sid(self) -> u8 {
        (self.xm & 15) as u8
    }

    pub const fn burst_count(self) -> u16 {
        ((self.xm >> 4) & 4095) as u16
    }

    pub const fn burst_bytes(self) -> u32 {
        ((self.xm >> 16) & 0x1f_ffff) as u32
    }

    pub const fn is_disabled(self) -> bool {
        self.burst_count() == 0 || self.burst_bytes() == 0
    }

    pub const fn is_input(self) -> bool {
        matches!(self.instruction.direction, C220MovDirection::HbmToUb)
    }

    pub const fn left_padding(self) -> u32 {
        if self.is_input() {
            ((self.xm >> 48) & 63) as u32
        } else {
            0
        }
    }

    pub const fn right_padding(self) -> u32 {
        if self.is_input() {
            ((self.xm >> 54) & 63) as u32
        } else {
            0
        }
    }

    pub const fn source_gap(self) -> u32 {
        self.xt as u32
    }

    pub const fn destination_gap(self) -> u32 {
        (self.xt >> 32) as u32
    }

    pub const fn padded_bytes(self) -> u32 {
        self.burst_bytes()
            + (self.left_padding() + self.right_padding()) * self.instruction.element_bytes as u32
    }

    pub const fn output_bytes(self) -> u32 {
        if self.is_input() {
            self.padded_bytes().div_ceil(32) * 32
        } else {
            self.burst_bytes()
        }
    }

    pub const fn source_stride(self) -> u32 {
        if self.is_input() {
            self.burst_bytes().wrapping_add(self.source_gap())
        } else {
            (self.burst_bytes().div_ceil(32) * 32).wrapping_add(self.source_gap().wrapping_mul(32))
        }
    }

    pub const fn destination_stride(self) -> u64 {
        if self.is_input() {
            self.output_bytes() as u64 + self.destination_gap() as u64 * 32
        } else {
            self.burst_bytes() as u64 + self.destination_gap() as u64
        }
    }

    pub fn segments(self) -> impl ExactSizeIterator<Item = C220MovPadSegment> {
        let count = if self.is_disabled() {
            0
        } else {
            self.burst_count()
        };
        (0..count).map(move |burst_index| C220MovPadSegment {
            burst_index,
            source_address: self.source_base.wrapping_add(u64::from(
                u32::from(burst_index).wrapping_mul(self.source_stride()),
            )),
            destination_address: self
                .destination_base
                .wrapping_add(u64::from(burst_index) * self.destination_stride()),
            input_bytes: self.burst_bytes(),
            output_bytes: self.output_bytes(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovPadSegment {
    pub burst_index: u16,
    pub source_address: u64,
    pub destination_address: u64,
    pub input_bytes: u32,
    pub output_bytes: u32,
}
