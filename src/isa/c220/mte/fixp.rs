/// Captured FIX operand registers. Decoding does not access execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpDescriptor {
    pub xt: u64,
    pub xm: u64,
    pub nd: u64,
}

impl C220FixpDescriptor {
    pub const fn stream_id(self) -> u8 {
        (self.xt & 15) as u8
    }
    pub const fn columns(self) -> u16 {
        ((self.xt >> 4) & 4095) as u16
    }
    pub const fn rows(self) -> u16 {
        (self.xt >> 16) as u16
    }
    pub const fn destination_stride(self) -> u32 {
        (self.xt >> 32) as u32
    }
    pub const fn source_stride(self) -> u16 {
        self.xm as u16
    }
    pub const fn unit_flag_mode(self) -> u8 {
        ((self.xm >> 32) & 3) as u8
    }
    pub const fn conversion_mode(self) -> u8 {
        ((self.xm >> 34) & 31) as u8
    }
    pub const fn activation_mode(self) -> u8 {
        ((self.xm >> 39) & 7) as u8
    }
    pub const fn channel_split(self) -> bool {
        self.xm & (1 << 42) != 0
    }
    pub const fn nz_to_nd(self) -> bool {
        self.xm & (1 << 43) != 0
    }
    pub const fn nd_count(self) -> u16 {
        self.nd as u16
    }
    pub const fn source_nd_stride(self) -> u16 {
        (self.nd >> 16) as u16
    }
    pub const fn destination_nd_stride(self) -> u32 {
        ((self.nd >> 32) & 0x1ffff) as u32
    }
    pub const fn is_disabled(self) -> bool {
        self.rows() == 0 || self.columns() == 0 || (self.nz_to_nd() && self.nd_count() == 0)
    }
}
