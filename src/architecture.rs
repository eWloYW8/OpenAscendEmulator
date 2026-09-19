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
    pub const fn simulator_directory(self) -> &'static str {
        match self {
            Self::Dav2201 => "dav_2201",
            Self::Dav3510 => "dav_3510",
        }
    }

    pub const fn default_soc_version(self) -> &'static str {
        match self {
            Self::Dav2201 => "Ascend910B1",
            Self::Dav3510 => "Ascend950PR_9599",
        }
    }

    pub const fn simulator_aic_cores(self) -> u16 {
        match self {
            Self::Dav2201 => 24,
            Self::Dav3510 => 32,
        }
    }

    pub const fn simulator_aiv_cores(self) -> u16 {
        match self {
            Self::Dav2201 => 48,
            Self::Dav3510 => 64,
        }
    }
}

impl fmt::Display for Architecture {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.simulator_directory())
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceProfile {
    pub soc_version: &'static str,
    pub architecture: Architecture,
    pub simulator_aic_cores: u16,
    pub simulator_aiv_cores: u16,
}

impl DeviceProfile {
    pub const fn new(soc_version: &'static str, architecture: Architecture) -> Self {
        Self {
            soc_version,
            architecture,
            simulator_aic_cores: architecture.simulator_aic_cores(),
            simulator_aiv_cores: architecture.simulator_aiv_cores(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedTarget {
    pub requested: String,
    pub camodel_soc_version: String,
    pub architecture: Architecture,
    pub simulator_directory: &'static str,
    pub simulator_aic_cores: u16,
    pub simulator_aiv_cores: u16,
}

impl ResolvedTarget {
    fn from_parts(requested: &str, camodel_soc_version: &str, architecture: Architecture) -> Self {
        Self {
            requested: requested.to_owned(),
            camodel_soc_version: camodel_soc_version.to_owned(),
            architecture,
            simulator_directory: architecture.simulator_directory(),
            simulator_aic_cores: architecture.simulator_aic_cores(),
            simulator_aiv_cores: architecture.simulator_aiv_cores(),
        }
    }
}

pub fn all_device_profiles() -> impl Iterator<Item = DeviceProfile> {
    DAV_2201_DEVICES
        .iter()
        .copied()
        .map(|soc| DeviceProfile::new(soc, Architecture::Dav2201))
        .chain(
            DAV_3510_DEVICES
                .iter()
                .copied()
                .map(|soc| DeviceProfile::new(soc, Architecture::Dav3510)),
        )
}

pub fn resolve_target(requested: Option<&str>) -> Option<ResolvedTarget> {
    let requested = requested.unwrap_or(Architecture::Dav2201.default_soc_version());
    match requested {
        "dav_2201" | "Ascend910B" => Some(ResolvedTarget::from_parts(
            requested,
            Architecture::Dav2201.default_soc_version(),
            Architecture::Dav2201,
        )),
        "dav_3510" | "Ascend950" => Some(ResolvedTarget::from_parts(
            requested,
            Architecture::Dav3510.default_soc_version(),
            Architecture::Dav3510,
        )),
        soc if DAV_2201_DEVICES.contains(&soc) => {
            Some(ResolvedTarget::from_parts(soc, soc, Architecture::Dav2201))
        }
        soc if DAV_3510_DEVICES.contains(&soc) => {
            Some(ResolvedTarget::from_parts(soc, soc, Architecture::Dav3510))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_contains_all_supported_devices() {
        let profiles: Vec<_> = all_device_profiles().collect();
        assert_eq!(profiles.len(), 38);
        assert_eq!(
            profiles
                .iter()
                .filter(|profile| profile.architecture == Architecture::Dav2201)
                .count(),
            6
        );
        assert_eq!(
            profiles
                .iter()
                .filter(|profile| profile.architecture == Architecture::Dav3510)
                .count(),
            32
        );
    }

    #[test]
    fn architecture_aliases_resolve_to_reference_defaults() {
        let c220 = resolve_target(Some("dav_2201")).unwrap();
        assert_eq!(c220.camodel_soc_version, "Ascend910B1");
        assert_eq!(c220.simulator_aic_cores, 24);
        assert_eq!(c220.simulator_aiv_cores, 48);

        let c310 = resolve_target(Some("dav_3510")).unwrap();
        assert_eq!(c310.camodel_soc_version, "Ascend950PR_9599");
        assert_eq!(c310.simulator_aic_cores, 32);
        assert_eq!(c310.simulator_aiv_cores, 64);
    }

    #[test]
    fn rejects_unknown_soc() {
        assert!(resolve_target(Some("Ascend9999")).is_none());
    }
}
