use super::load2d::{C220_LOAD_2D_BLOCK_BYTES, C220Load2dDestination, C220Load2dElementFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dTransposeInstruction {
    pub word: u32,
    pub destination: C220Load2dDestination,
    pub element_format: C220Load2dElementFormat,
    pub destination_register: u8,
    pub source_register: u8,
    pub descriptor_register: u8,
    pub stride_register: u8,
}

impl C220Load2dTransposeInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 1 || (word >> 24) & 7 != 5 {
            return None;
        }
        Some(Self {
            word,
            destination: if word & 1 == 0 {
                C220Load2dDestination::L0a
            } else {
                C220Load2dDestination::L0b
            },
            element_format: match (word >> 22) & 3 {
                0 => C220Load2dElementFormat::B8,
                1 => C220Load2dElementFormat::B16,
                2 => C220Load2dElementFormat::B4,
                _ => C220Load2dElementFormat::B32,
            },
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            descriptor_register: ((word >> 7) & 31) as u8,
            stride_register: ((word >> 2) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220Load2dTransposeTransfer {
        C220Load2dTransposeTransfer {
            instruction: self,
            source_base: registers[usize::from(self.source_register)],
            destination_base: registers[usize::from(self.destination_register)],
            xm: registers[usize::from(self.descriptor_register)],
            xt: registers[usize::from(self.stride_register)],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dTransposeTransfer {
    pub instruction: C220Load2dTransposeInstruction,
    pub source_base: u64,
    pub destination_base: u64,
    pub xm: u64,
    pub xt: u64,
}

impl C220Load2dTransposeTransfer {
    pub const fn repeat_count(self) -> u8 {
        (self.xm >> 16) as u8
    }

    pub const fn fractals_per_repeat(self) -> u8 {
        match self.instruction.element_format {
            C220Load2dElementFormat::B4 => 4,
            C220Load2dElementFormat::B8 | C220Load2dElementFormat::B32 => 2,
            C220Load2dElementFormat::B16 => 1,
        }
    }

    pub const fn start_index(self) -> u16 {
        self.xm as u16
    }

    pub const fn source_stride(self) -> u16 {
        (self.xm >> 24) as u16
    }

    pub const fn destination_stride(self) -> u32 {
        (self.xm >> 44) as u16 as u32 + 1
    }

    pub const fn destination_fractal_stride(self) -> u32 {
        self.xt as u16 as u32 + 1
    }

    pub const fn decrement(self) -> bool {
        self.xm >> 63 != 0
    }

    pub fn segments(self) -> impl ExactSizeIterator<Item = C220Load2dTransposeSegment> + Clone {
        let group = u32::from(self.fractals_per_repeat());
        (0..u16::from(self.repeat_count()) * u16::from(self.fractals_per_repeat())).map(
            move |index| {
                let repeat = u32::from(index) / group;
                let fractal = u32::from(index) % group;
                let offset = repeat.wrapping_mul(u32::from(self.source_stride()));
                let source_index = if self.decrement() {
                    u32::from(self.start_index()).wrapping_sub(offset)
                } else {
                    u32::from(self.start_index()).wrapping_add(offset)
                };
                let source_offset = source_index
                    .wrapping_mul(group)
                    .wrapping_add(fractal)
                    .wrapping_mul(C220_LOAD_2D_BLOCK_BYTES);
                let destination_offset = (u64::from(repeat) * u64::from(self.destination_stride())
                    + u64::from(fractal.wrapping_mul(self.destination_fractal_stride())))
                .wrapping_mul(u64::from(C220_LOAD_2D_BLOCK_BYTES));
                C220Load2dTransposeSegment {
                    repeat_index: repeat as u8,
                    fractal_index: fractal as u8,
                    source_address: self.source_base.wrapping_add(u64::from(source_offset)),
                    destination_address: self.destination_base.wrapping_add(destination_offset),
                    bytes: C220_LOAD_2D_BLOCK_BYTES,
                }
            },
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load2dTransposeSegment {
    pub repeat_index: u8,
    pub fractal_index: u8,
    pub source_address: u64,
    pub destination_address: u64,
    pub bytes: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_addresses_capture_registers_and_wrap_source_offsets() {
        for (format, group) in [(0, 2), (1, 1), (2, 4), (3, 2)] {
            for destination in 0..=1 {
                let word = (3 << 29)
                    | (1 << 27)
                    | (5 << 24)
                    | (format << 22)
                    | (2 << 12)
                    | (3 << 7)
                    | (4 << 2)
                    | destination;
                let instruction = C220Load2dTransposeInstruction::decode(word).unwrap();
                let mut registers = [0; 32];
                registers[0] = 64;
                registers[2] = 128;
                registers[3] = (1 << 63) | (2 << 16) | (1 << 24) | (7 << 44);
                registers[4] = 2;
                let transfer = instruction.capture(&registers);
                assert_eq!(transfer.fractals_per_repeat(), group);
                let segments = transfer.segments().collect::<Vec<_>>();
                assert_eq!(segments.len(), 2 * usize::from(group));
                for (index, segment) in segments.iter().enumerate() {
                    let repeat = index / usize::from(group);
                    let fractal = index % usize::from(group);
                    let source = if repeat == 0 {
                        128
                    } else {
                        128 + (1_u64 << 32) - u64::from(group) * 512
                    } + fractal as u64 * 512;
                    assert_eq!(segment.source_address, source);
                    assert_eq!(
                        segment.destination_address,
                        64 + (repeat * 8 + fractal * 3) as u64 * 512
                    );
                }
                registers[3] = 0;
                assert_eq!(instruction.capture(&registers).segments().len(), 0);
            }
        }
        assert!(C220Load2dTransposeInstruction::decode(0x6000_2184).is_none());
    }
}
