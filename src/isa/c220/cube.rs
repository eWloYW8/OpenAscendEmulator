pub const C220_CUBE_ARRAY_EDGE: u16 = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeOperation {
    Mmad,
    SparseMmad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeDataType {
    S4S4S32,
    U8U8S32,
    S8S8S32,
    F16F16,
    F16F32,
    F16U2,
    U8S8S32,
    B8U2,
    Bf16F32,
    F32F32,
}

impl C220CubeDataType {
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::U8U8S32),
            1 => Some(Self::S8S8S32),
            2 => Some(Self::F16F16),
            3 => Some(Self::F16F32),
            4 => Some(Self::F16U2),
            5 => Some(Self::U8S8S32),
            6 => Some(Self::S4S4S32),
            7 => Some(Self::B8U2),
            9 => Some(Self::Bf16F32),
            10 => Some(Self::F32F32),
            _ => None,
        }
    }

    pub const fn k_tile_elements(self) -> u16 {
        match self {
            Self::S4S4S32 => 64,
            Self::U8U8S32 | Self::S8S8S32 | Self::U8S8S32 | Self::B8U2 => 32,
            Self::F32F32 => 8,
            Self::F16F16 | Self::F16F32 | Self::F16U2 | Self::Bf16F32 => 16,
        }
    }

    pub const fn l0c_request_bytes(self) -> u16 {
        match self {
            Self::F16F16 | Self::F16U2 => 512,
            _ => 1024,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeInstruction {
    pub word: u32,
    pub operation: C220CubeOperation,
    pub data_type: C220CubeDataType,
    pub raw_data_type: u8,
    pub xd: u8,
    pub xn: u8,
    pub xm: u8,
    pub xt: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeRegisterValues {
    pub xd: u64,
    pub xn: u64,
    pub xm: u64,
    pub xt: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MmadParameters {
    pub xd_low: u32,
    pub xd_high: u32,
    pub xn: u64,
    pub xm: u64,
    pub m: u16,
    pub raw_k: u16,
    pub effective_k: u16,
    pub n: u16,
    pub xt_bits_44_50: u8,
    pub xt_bits_55_56: u8,
    pub xt_bit_58: bool,
    pub xt_bit_62: bool,
    pub xt_bit_63: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeGeometry {
    pub m_tiles: u16,
    pub n_tiles: u16,
    pub k_tiles: u16,
    pub k_tile_elements: u16,
    pub native_uop_count: u64,
}

impl C220CubeInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 7 {
            return None;
        }
        let operation = match (word >> 25) & 7 {
            0 => C220CubeOperation::Mmad,
            5 => C220CubeOperation::SparseMmad,
            _ => return None,
        };
        let raw_data_type = (((word >> 22) & 7) | ((word & 1) << 3)) as u8;
        let Some(data_type) = C220CubeDataType::from_raw(raw_data_type) else {
            return None;
        };
        Some(Self {
            word,
            operation,
            data_type,
            raw_data_type,
            xd: ((word >> 17) & 0x1f) as u8,
            xn: ((word >> 12) & 0x1f) as u8,
            xm: ((word >> 7) & 0x1f) as u8,
            xt: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn capture(self, xregs: &[u64; 32]) -> C220CubeRegisterValues {
        C220CubeRegisterValues {
            xd: xregs[self.xd as usize],
            xn: xregs[self.xn as usize],
            xm: xregs[self.xm as usize],
            xt: xregs[self.xt as usize],
        }
    }

    pub const fn parameters(self, registers: C220CubeRegisterValues) -> C220MmadParameters {
        let m = (registers.xt & 0xfff) as u16;
        let raw_k = ((registers.xt >> 12) & 0xfff) as u16;
        let n = ((registers.xt >> 24) & 0xfff) as u16;
        let effective_k = match self.operation {
            C220CubeOperation::Mmad => raw_k,
            C220CubeOperation::SparseMmad => raw_k.div_ceil(4) * 2,
        };
        C220MmadParameters {
            xd_low: registers.xd as u32,
            xd_high: (registers.xd >> 32) as u32,
            xn: registers.xn,
            xm: registers.xm,
            m,
            raw_k,
            effective_k,
            n,
            xt_bits_44_50: ((registers.xt >> 44) & 0x7f) as u8,
            xt_bits_55_56: ((registers.xt >> 55) & 3) as u8,
            xt_bit_58: registers.xt & (1 << 58) != 0,
            xt_bit_62: registers.xt & (1 << 62) != 0,
            xt_bit_63: registers.xt & (1 << 63) != 0,
        }
    }

    pub const fn geometry(self, parameters: C220MmadParameters) -> C220CubeGeometry {
        let k_tile_elements = self.data_type.k_tile_elements();
        let m_tiles = parameters.m.div_ceil(C220_CUBE_ARRAY_EDGE);
        let n_tiles = parameters.n.div_ceil(C220_CUBE_ARRAY_EDGE);
        let k_tiles = parameters.effective_k.div_ceil(k_tile_elements);
        C220CubeGeometry {
            m_tiles,
            n_tiles,
            k_tiles,
            k_tile_elements,
            native_uop_count: m_tiles as u64 * n_tiles as u64 * k_tiles as u64,
        }
    }
}
