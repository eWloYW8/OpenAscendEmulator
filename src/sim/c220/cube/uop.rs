#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeL0cAccess {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeUnitFlagMode {
    Disabled,
    Check,
    CheckAndUpdate,
}

impl C220CubeUnitFlagMode {
    pub const fn from_xt_bits(bits: u8) -> Self {
        match bits & 0b11 {
            2 => Self::Check,
            3 => Self::CheckAndUpdate,
            _ => Self::Disabled,
        }
    }

    pub const fn checks(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    pub const fn updates(self) -> bool {
        matches!(self, Self::CheckAndUpdate)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeL0cRequest {
    pub address: u64,
    pub bytes: u16,
    pub access: C220CubeL0cAccess,
    pub unit_flags: C220CubeUnitFlagMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeTileIndices {
    pub l0a: u32,
    pub l0b: u32,
    pub l0c: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeUop {
    pub id: u64,
    pub pre_issue_bubbles: u8,
    pub tile_indices: Option<C220CubeTileIndices>,
    pub reads_l0a: bool,
    pub reads_l0b: bool,
    pub acquires_l0c_write_port: bool,
    pub l0c_read: Option<C220CubeL0cRequest>,
    pub l0c_write: Option<C220CubeL0cRequest>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeUopRelease {
    pub instruction_id: u64,
    pub uop: C220CubeUop,
    pub issue_tick: u64,
}
