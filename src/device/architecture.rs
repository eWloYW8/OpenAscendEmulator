use serde::Serialize;
use std::fmt;
use std::str::FromStr;

pub const DAV_2201_DEVICES: &[&str] = &[
    "Ascend910B1",
    "Ascend910B2",
    "Ascend910B3",
    "Ascend910B4",
    "Ascend910B2C",
    "Ascend910B4-1",
];

pub const DAV_3510_DEVICES: &[&str] = &[
    "Ascend950DT_950x",
    "Ascend950DT_950y",
    "Ascend950DT_9571",
    "Ascend950DT_9572",
    "Ascend950DT_9573",
    "Ascend950DT_9574",
    "Ascend950DT_9575",
    "Ascend950DT_9576",
    "Ascend950DT_9577",
    "Ascend950DT_9578",
    "Ascend950DT_9581",
    "Ascend950DT_9582",
    "Ascend950DT_9583",
    "Ascend950DT_9584",
    "Ascend950DT_9585",
    "Ascend950DT_9586",
    "Ascend950DT_9587",
    "Ascend950DT_9588",
    "Ascend950DT_9591",
    "Ascend950DT_9592",
    "Ascend950DT_9595",
    "Ascend950DT_9596",
    "Ascend950DT_95A1",
    "Ascend950DT_95A2",
    "Ascend950PR_950z",
    "Ascend950PR_9579",
    "Ascend950PR_957b",
    "Ascend950PR_957c",
    "Ascend950PR_957d",
    "Ascend950PR_9589",
    "Ascend950PR_958b",
    "Ascend950PR_9599",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    #[serde(rename = "dav_2201")]
    Dav2201,
    #[serde(rename = "dav_3510")]
    Dav3510,
}

impl Architecture {
    pub fn from_device(name: &str) -> Option<Self> {
        if DAV_2201_DEVICES.contains(&name) {
            Some(Self::Dav2201)
        } else if DAV_3510_DEVICES.contains(&name) {
            Some(Self::Dav3510)
        } else {
            None
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_all_supported_devices() {
        assert_eq!(DAV_2201_DEVICES.len(), 6);
        assert_eq!(DAV_3510_DEVICES.len(), 32);
        assert!(
            DAV_2201_DEVICES
                .iter()
                .all(|device| Architecture::from_device(device) == Some(Architecture::Dav2201))
        );
        assert!(
            DAV_3510_DEVICES
                .iter()
                .all(|device| Architecture::from_device(device) == Some(Architecture::Dav3510))
        );
    }

    #[test]
    fn unknown_device_has_no_architecture() {
        assert_eq!(Architecture::from_device("Ascend9999"), None);
    }
}
