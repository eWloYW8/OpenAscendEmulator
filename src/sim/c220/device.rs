use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum C220Device {
    #[default]
    Ascend910B1,
    Ascend910B2,
    Ascend910B3,
    Ascend910B4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220DeviceProfile {
    pub cube_cores: u16,
    pub vector_cores: u16,
    pub cube_frequency_mhz: u16,
    pub hbm_bytes: u64,
    pub l2_bytes: u64,
    pub l0a_bytes: u32,
    pub l0b_bytes: u32,
    pub l0c_bytes: u32,
    pub l1_bytes: u32,
    pub ub_bytes: u32,
}

impl C220Device {
    pub const fn profile(self) -> C220DeviceProfile {
        let (cube_cores, vector_cores, cube_frequency_mhz, hbm_bytes, l2_bytes) = match self {
            Self::Ascend910B1 => (24, 48, 1850, 68_719_476_736, 201_326_592),
            Self::Ascend910B2 => (24, 48, 1800, 68_719_476_736, 201_326_592),
            Self::Ascend910B3 => (20, 40, 1800, 68_719_476_736, 201_326_592),
            Self::Ascend910B4 => (20, 40, 1500, 34_359_738_368, 100_663_296),
        };
        C220DeviceProfile {
            cube_cores,
            vector_cores,
            cube_frequency_mhz,
            hbm_bytes,
            l2_bytes,
            l0a_bytes: 65_536,
            l0b_bytes: 65_536,
            l0c_bytes: 131_072,
            l1_bytes: 524_288,
            ub_bytes: 196_608,
        }
    }
}

impl fmt::Display for C220Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Ascend910B1 => "Ascend910B1",
            Self::Ascend910B2 => "Ascend910B2",
            Self::Ascend910B3 => "Ascend910B3",
            Self::Ascend910B4 => "Ascend910B4",
        })
    }
}

impl FromStr for C220Device {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "ascend910b1" | "910b1" => Ok(Self::Ascend910B1),
            "ascend910b2" | "910b2" => Ok(Self::Ascend910B2),
            "ascend910b3" | "910b3" => Ok(Self::Ascend910B3),
            "ascend910b4" | "910b4" => Ok(Self::Ascend910B4),
            _ => Err(format!("unsupported C220 device: {value}")),
        }
    }
}
