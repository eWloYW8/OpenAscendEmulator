use std::fmt;
use std::str::FromStr;

pub mod c220;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Architecture {
    Dav2201,
    Dav3510,
}

impl fmt::Display for Architecture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Dav2201 => "dav_2201",
            Self::Dav3510 => "dav_3510",
        })
    }
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "dav_2201" => Ok(Self::Dav2201),
            "dav_3510" => Ok(Self::Dav3510),
            _ => Err(format!("unsupported architecture: {value}")),
        }
    }
}
