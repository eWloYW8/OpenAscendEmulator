#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load3dDestination {
    L0a,
    L0b,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220Load3dElement {
    B4,
    B8,
    B16,
    B32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dV2Instruction {
    pub word: u32,
    pub destination: C220Load3dDestination,
    pub element: C220Load3dElement,
    pub destination_register: u8,
    pub source_register: u8,
    pub extent_register: u8,
    pub geometry_register: u8,
}

impl C220Load3dV2Instruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 3 || (word >> 27) & 3 != 0 {
            return None;
        }
        if !matches!((word >> 22) & 31, 20..=23 | 28..=31) {
            return None;
        }
        Some(Self {
            word,
            destination: if word & ((1 << 1) | (1 << 23)) != 0 {
                C220Load3dDestination::L0b
            } else {
                C220Load3dDestination::L0a
            },
            element: match ((word >> 22) & 1) | (((word >> 25) & 1) << 1) {
                0 => C220Load3dElement::B8,
                1 => C220Load3dElement::B16,
                2 => C220Load3dElement::B4,
                _ => C220Load3dElement::B32,
            },
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            extent_register: ((word >> 7) & 31) as u8,
            geometry_register: ((word >> 2) & 31) as u8,
        })
    }

    pub fn capture(self, registers: &[u64; 32]) -> C220Load3dV2Operands {
        C220Load3dV2Operands {
            instruction: self,
            destination_base: registers[usize::from(self.destination_register)],
            source_base: registers[usize::from(self.source_register)],
            extent: C220Load3dExtent::decode(registers[usize::from(self.extent_register)]),
            geometry: C220Load3dGeometry::decode(registers[usize::from(self.geometry_register)]),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dV2Operands {
    pub instruction: C220Load3dV2Instruction,
    pub destination_base: u64,
    pub source_base: u64,
    pub extent: C220Load3dExtent,
    pub geometry: C220Load3dGeometry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dExtent {
    pub raw: u64,
    pub k_length: u16,
    pub m_length: u16,
    pub k_start: u16,
    pub m_start: u16,
}

impl C220Load3dExtent {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            k_length: raw as u16,
            m_length: (raw >> 16) as u16,
            k_start: (raw >> 32) as u16,
            m_start: (raw >> 48) as u16,
        }
    }
}

/// Encoded geometry. Execution normalizes zero strides and dilations separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dGeometry {
    pub raw: u64,
    pub stride_w: u8,
    pub stride_h: u8,
    pub filter_w: u16,
    pub filter_h: u16,
    pub dilation_w: u8,
    pub dilation_h: u8,
    pub transpose: bool,
    pub alternate_matrix: bool,
    pub channel_size: u16,
}

impl C220Load3dGeometry {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            stride_w: (raw & 63) as u8,
            stride_h: ((raw >> 6) & 63) as u8,
            filter_w: (((raw >> 12) & 255) | (((raw >> 44) & 1) << 8)) as u16,
            filter_h: (((raw >> 20) & 255) | (((raw >> 45) & 1) << 8)) as u16,
            dilation_w: (raw >> 28) as u8,
            dilation_h: (raw >> 36) as u8,
            transpose: raw & (1 << 46) != 0,
            alternate_matrix: raw & (1 << 47) != 0,
            channel_size: (raw >> 48) as u16,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dMatrix {
    pub raw: u64,
    pub width: u16,
    pub height: u16,
    pub pad_left: u8,
    pub pad_right: u8,
    pub pad_top: u8,
    pub pad_bottom: u8,
}

impl C220Load3dMatrix {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            width: raw as u16,
            height: (raw >> 16) as u16,
            pad_left: (raw >> 32) as u8,
            pad_right: (raw >> 40) as u8,
            pad_top: (raw >> 48) as u8,
            pad_bottom: (raw >> 56) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dRepeat {
    pub raw: u64,
    pub stride: u16,
    pub count: u8,
    pub k_mode: bool,
}

impl C220Load3dRepeat {
    pub const fn decode(raw: u64) -> Self {
        Self {
            raw,
            stride: raw as u16,
            count: (raw >> 16) as u8,
            k_mode: raw & (1 << 24) != 0,
        }
    }
}
