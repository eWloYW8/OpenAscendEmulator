#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220AxpyWidth {
    F16,
    F16ToF32,
    F32,
}

impl C220AxpyWidth {
    pub const fn source_element_bytes(self) -> u8 {
        match self {
            Self::F16 | Self::F16ToF32 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn destination_element_bytes(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F16ToF32 | Self::F32 => 4,
        }
    }

    pub const fn lane_count(self) -> usize {
        256 / self.destination_element_bytes() as usize
    }

    pub const fn lane_groups(self) -> u8 {
        match self {
            Self::F16 => 2,
            Self::F16ToF32 | Self::F32 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220AxpyInstruction {
    pub width: C220AxpyWidth,
    pub destination_register: u8,
    pub source_register: u8,
    pub scalar_register: u8,
    pub control_register: u8,
}

impl C220AxpyInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let width = match word & 0xffc0_0003 {
            0x9540_0000 => C220AxpyWidth::F16,
            0x9580_0000 => C220AxpyWidth::F16ToF32,
            0x95c0_0000 => C220AxpyWidth::F32,
            _ => return None,
        };
        Some(Self {
            width,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            scalar_register: ((word >> 7) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }
}
