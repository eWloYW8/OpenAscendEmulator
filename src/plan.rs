use crate::architecture::{ResolvedTarget, resolve_target};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SimulatorRequest {
    pub config: Option<PathBuf>,
    pub application: Option<PathBuf>,
    pub export: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub kernel_name: Option<String>,
    pub aic_metrics: Vec<String>,
    pub launch_count: Option<u32>,
    pub mstx: Option<bool>,
    pub mstx_include: Option<String>,
    pub soc_version: Option<String>,
    pub core_ids: Vec<u32>,
    pub timeout_minutes: Option<u32>,
    pub dump: Option<bool>,
    pub application_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlanPaths {
    pub ascend_home: PathBuf,
    pub simulator_root: PathBuf,
    pub simulator_library_directory: PathBuf,
    pub injection_library: PathBuf,
    pub kernel_launcher: PathBuf,
    pub output_directory: PathBuf,
    pub simulator_dump_directory: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandSpec {
    pub executable: PathBuf,
    pub arguments: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LaunchPlan {
    pub target: ResolvedTarget,
    pub paths: PlanPaths,
    pub environment: BTreeMap<String, String>,
    pub command: CommandSpec,
    pub request: SimulatorRequest,
    pub unresolved: Vec<&'static str>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    #[error("unsupported --soc-version: {0}")]
    UnsupportedSoc(String),
    #[error("--soc-version is not effective in config mode")]
    SocVersionInConfigMode,
    #[error("--launch-count must be in [1, 5000]")]
    InvalidLaunchCount,
    #[error("--timeout must be in [1, 2880]")]
    InvalidTimeout,
    #[error("PipeUtilization is required when simulator AIC metrics are requested")]
    MissingPipeUtilization,
}

impl LaunchPlan {
    pub fn build(
        request: SimulatorRequest,
        ascend_home: impl AsRef<Path>,
        inherited_ld_library_path: Option<&str>,
    ) -> Result<Self, PlanError> {
        if request.config.is_some() && request.soc_version.is_some() {
            return Err(PlanError::SocVersionInConfigMode);
        }
        if request
            .launch_count
            .is_some_and(|count| !(1..=5000).contains(&count))
        {
            return Err(PlanError::InvalidLaunchCount);
        }
        if request
            .timeout_minutes
            .is_some_and(|minutes| !(1..=2880).contains(&minutes))
        {
            return Err(PlanError::InvalidTimeout);
        }
        if !request.aic_metrics.is_empty()
            && !request
                .aic_metrics
                .iter()
                .any(|metric| metric.eq_ignore_ascii_case("PipeUtilization"))
        {
            return Err(PlanError::MissingPipeUtilization);
        }

        let target = resolve_target(request.soc_version.as_deref()).ok_or_else(|| {
            PlanError::UnsupportedSoc(request.soc_version.clone().unwrap_or_default())
        })?;
        let ascend_home = ascend_home.as_ref().to_path_buf();
        let simulator_root = ascend_home.join("tools/simulator");
        let simulator_library_directory =
            simulator_root.join(target.simulator_directory).join("lib");
        let injection_library = ascend_home.join("tools/msopprof/lib64/libmsopprof_injection.so");
        let kernel_launcher = ascend_home.join("tools/msopprof/bin/kernel-launcher");
        let output_directory = request
            .output
            .clone()
            .unwrap_or_else(|| PathBuf::from("./output"));
        let simulator_dump_directory = output_directory.join("device0/tmp_dump");

        let mut environment = BTreeMap::new();
        let mut ld_library_path = simulator_library_directory.display().to_string();
        if let Some(inherited) = inherited_ld_library_path.filter(|value| !value.is_empty()) {
            ld_library_path.push(':');
            ld_library_path.push_str(inherited);
        }
        environment.insert("LD_LIBRARY_PATH".into(), ld_library_path);
        environment.insert(
            "LD_PRELOAD".into(),
            format!("{}:libruntime_camodel.so", injection_library.display()),
        );
        environment.insert(
            "CAMODEL_SOC_VERSION".into(),
            target.camodel_soc_version.clone(),
        );
        environment.insert(
            "CAMODEL_LOG_PATH".into(),
            simulator_dump_directory.display().to_string(),
        );
        environment.insert("IS_SIMULATOR_ENV".into(), "true".into());
        environment.insert("TASK_QUEUE_ENABLE".into(), "0".into());
        environment.insert("GE_INIT_DISABLE".into(), "1".into());
        environment.insert("ENABLE_CA_LOG_TRANS".into(), "true".into());

        let command = if let Some(application) = request.application.clone() {
            CommandSpec {
                executable: application,
                arguments: request.application_args.clone(),
            }
        } else {
            CommandSpec {
                executable: kernel_launcher.clone(),
                arguments: Vec::new(),
            }
        };

        Ok(Self {
            target,
            paths: PlanPaths {
                ascend_home,
                simulator_root,
                simulator_library_directory,
                injection_library,
                kernel_launcher,
                output_directory,
                simulator_dump_directory,
            },
            environment,
            command,
            request,
            unresolved: vec![
                "kernel replay artifact persistence and ACL execution",
                "kernel-record IPC body schema and local-process transport backend",
                "instruction timing model and report serialization",
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c220_plan_sets_simulator_environment() {
        let request = SimulatorRequest {
            application: Some(PathBuf::from("./operator")),
            soc_version: Some("Ascend910B4-1".into()),
            ..SimulatorRequest::default()
        };
        let plan = LaunchPlan::build(request, "/opt/ascend", Some("/host/lib")).unwrap();
        assert_eq!(plan.target.architecture.to_string(), "dav_2201");
        assert_eq!(plan.environment["CAMODEL_SOC_VERSION"], "Ascend910B4-1");
        assert_eq!(
            plan.environment["LD_LIBRARY_PATH"],
            "/opt/ascend/tools/simulator/dav_2201/lib:/host/lib"
        );
        assert_eq!(
            plan.environment["LD_PRELOAD"],
            "/opt/ascend/tools/msopprof/lib64/libmsopprof_injection.so:libruntime_camodel.so"
        );
    }

    #[test]
    fn c310_alias_uses_default_soc() {
        let request = SimulatorRequest {
            soc_version: Some("dav_3510".into()),
            ..SimulatorRequest::default()
        };
        let plan = LaunchPlan::build(request, "/opt/ascend", None).unwrap();
        assert_eq!(plan.target.camodel_soc_version, "Ascend950PR_9599");
        assert_eq!(plan.environment["IS_SIMULATOR_ENV"], "true");
        assert_eq!(plan.environment["TASK_QUEUE_ENABLE"], "0");
        assert_eq!(plan.environment["GE_INIT_DISABLE"], "1");
    }

    #[test]
    fn rejects_reference_incompatible_combinations() {
        let request = SimulatorRequest {
            config: Some(PathBuf::from("case.json")),
            soc_version: Some("Ascend910B1".into()),
            ..SimulatorRequest::default()
        };
        assert_eq!(
            LaunchPlan::build(request, "/opt/ascend", None),
            Err(PlanError::SocVersionInConfigMode)
        );
    }
}
