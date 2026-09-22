#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220CubeL0cAccess {
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeL0cRequest {
    pub address: u64,
    pub bytes: u16,
    pub access: C220CubeL0cAccess,
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
    pub uop: C220CubeUop,
    pub issue_tick: u64,
}
