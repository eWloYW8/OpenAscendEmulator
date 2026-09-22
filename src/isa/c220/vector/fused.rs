#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FusedOperation {
    AddRelu,
    AddDeqRelu,
    SubtractRelu,
    Multiply,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FusedFormat {
    S16ToS8,
    F16ToS8,
    F16ToU8,
    F32ToF16,
    S32ToF16,
    VectorDeqS16ToB8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FusedInstruction {
    pub operation: C220FusedOperation,
    pub format: C220FusedFormat,
    pub destination_high: bool,
    pub destination_register: u8,
    pub source_0_register: u8,
    pub source_1_register: u8,
    pub control_register: u8,
}

impl C220FusedInstruction {
    pub fn decode(word: u32) -> Option<Self> {
        let (operation, format, destination_high) = if word & 0xff00_0002 == 0x8a00_0000 {
            (
                C220FusedOperation::AddRelu,
                decode_add_sub_format(((word >> 22) & 3) as u8),
                word & 1 != 0,
            )
        } else if word & 0xff00_0002 == 0x8a00_0002 {
            (
                C220FusedOperation::SubtractRelu,
                decode_add_sub_format(((word >> 22) & 3) as u8),
                word & 1 != 0,
            )
        } else if word & 0xfe00_0003 == 0x9e00_0001 {
            (
                C220FusedOperation::Multiply,
                if (word >> 22) & 3 == 1 {
                    C220FusedFormat::F16ToS8
                } else {
                    C220FusedFormat::F16ToU8
                },
                word & (1 << 24) != 0,
            )
        } else if word & 0xffc0_0003 == 0x9ec0_0003 {
            (
                C220FusedOperation::AddDeqRelu,
                C220FusedFormat::S32ToF16,
                false,
            )
        } else {
            return None;
        };
        Some(Self {
            operation,
            format,
            destination_high,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_0_register: ((word >> 12) & 0x1f) as u8,
            source_1_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn lane_count(self) -> usize {
        match self.format {
            C220FusedFormat::F32ToF16
            | C220FusedFormat::S32ToF16
            | C220FusedFormat::VectorDeqS16ToB8 => 64,
            C220FusedFormat::S16ToS8 | C220FusedFormat::F16ToS8 | C220FusedFormat::F16ToU8 => 128,
        }
    }

    pub const fn lane_groups(self) -> u8 {
        self.lane_count().div_ceil(64) as u8
    }

    pub const fn source_element_bytes(self) -> u8 {
        match self.format {
            C220FusedFormat::F32ToF16 | C220FusedFormat::S32ToF16 => 4,
            _ => 2,
        }
    }

    pub const fn destination_element_bytes(self) -> u8 {
        match self.format {
            C220FusedFormat::F32ToF16 | C220FusedFormat::S32ToF16 => 2,
            _ => 1,
        }
    }

    pub const fn execute_ticks(self) -> u8 {
        match (self.operation, self.format) {
            (C220FusedOperation::Multiply | C220FusedOperation::AddDeqRelu, _) => 10,
            (_, C220FusedFormat::S16ToS8) => 5,
            (_, C220FusedFormat::F16ToS8 | C220FusedFormat::F32ToF16) => 9,
            (_, C220FusedFormat::VectorDeqS16ToB8) => 11,
            (_, C220FusedFormat::F16ToU8 | C220FusedFormat::S32ToF16) => 10,
        }
    }
}

const fn decode_add_sub_format(selector: u8) -> C220FusedFormat {
    match selector {
        0 => C220FusedFormat::S16ToS8,
        1 => C220FusedFormat::F16ToS8,
        2 => C220FusedFormat::F32ToF16,
        _ => C220FusedFormat::VectorDeqS16ToB8,
    }
}
