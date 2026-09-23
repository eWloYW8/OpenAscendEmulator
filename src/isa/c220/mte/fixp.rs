#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220FixpDestination {
    External,
    L1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpInstruction {
    pub destination: C220FixpDestination,
    pub source_format: u8,
    pub destination_register: u8,
    pub source_register: u8,
    pub shape_register: u8,
    pub control_register: u8,
}

impl C220FixpInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if word >> 29 != 6 {
            return None;
        }
        let destination = match (word >> 24) & 31 {
            2 => C220FixpDestination::External,
            3 => C220FixpDestination::L1,
            _ => return None,
        };
        Some(Self {
            destination,
            source_format: (word & 3) as u8,
            destination_register: ((word >> 17) & 31) as u8,
            source_register: ((word >> 12) & 31) as u8,
            shape_register: ((word >> 7) & 31) as u8,
            control_register: ((word >> 2) & 31) as u8,
        })
    }
}

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
