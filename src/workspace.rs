use crate::plan::{CommandSpec, LaunchPlan};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use thiserror::Error;

const CONFIG_CANDIDATES: &[&str] = &[
    "config.json",
    "config_stars.json",
    "config_hwts.json",
    "Ascend910_93_model.toml",
    "pem_config_cloud.toml",
    "davinci_vec_core.spec",
    "davinci_mini.spec",
    "common.spec",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceOptions {
    pub enable_parallel_simulation: bool,
    pub parallel_thread_limit: u64,
    pub enable_cache: bool,
    pub flush_level: Option<u64>,
}

impl Default for WorkspaceOptions {
    fn default() -> Self {
        Self {
            enable_parallel_simulation: true,
            parallel_thread_limit: 24,
            enable_cache: true,
            flush_level: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ConfigPatch {
    pub file: PathBuf,
    pub field: String,
    pub before: String,
    pub after: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PreparedRun {
    pub root: PathBuf,
    pub config_directory: PathBuf,
    pub library_overlay: PathBuf,
    pub runtime_link: PathBuf,
    pub runtime_target: PathBuf,
    pub output_directory: PathBuf,
    pub simulator_dump_directory: PathBuf,
    pub copied_config_files: Vec<PathBuf>,
    pub config_patches: Vec<ConfigPatch>,
    pub environment: BTreeMap<String, String>,
    pub command: CommandSpec,
}

impl PreparedRun {
    pub fn prepare(
        plan: &LaunchPlan,
        root: impl AsRef<Path>,
        options: WorkspaceOptions,
    ) -> Result<Self, WorkspaceError> {
        let root = root.as_ref().to_path_buf();
        let config_directory = root.join("config");
        let library_overlay = root.join("lib");
        create_dir(&config_directory)?;
        create_dir(&library_overlay)?;
        create_dir(&plan.paths.output_directory)?;
        create_dir(&plan.paths.simulator_dump_directory)?;

        let runtime_target = plan
            .paths
            .simulator_library_directory
            .join("libruntime_camodel.so");
        if !runtime_target.is_file() {
            return Err(WorkspaceError::MissingRuntime(runtime_target));
        }
        let runtime_target = fs::canonicalize(&runtime_target)
            .map_err(|source| io_error("canonicalize", runtime_target.clone(), source))?;
        let runtime_link = library_overlay.join("libruntime.so");
        create_runtime_link(&runtime_link, &runtime_target)?;

        let mut copied_config_files = Vec::new();
        let mut config_patches = Vec::new();
        for name in CONFIG_CANDIDATES {
            let source = plan.paths.simulator_library_directory.join(name);
            if !source.is_file() {
                continue;
            }
            let destination = config_directory.join(name);
            if name.ends_with(".json") {
                copy_and_patch_json(&source, &destination, options, &mut config_patches)?;
            } else {
                copy_and_patch_text(&source, &destination, options, &mut config_patches)?;
            }
            copied_config_files.push(destination);
        }
        if copied_config_files.is_empty() {
            return Err(WorkspaceError::NoModelConfiguration(
                plan.paths.simulator_library_directory.clone(),
            ));
        }

        let mut environment = plan.environment.clone();
        environment.insert(
            "CAMODEL_CONFIG_PATH".into(),
            config_directory.display().to_string(),
        );
        let original_library_path = environment
            .get("LD_LIBRARY_PATH")
            .map(String::as_str)
            .unwrap_or_default();
        let library_path = if original_library_path.is_empty() {
            library_overlay.display().to_string()
        } else {
            format!("{}:{original_library_path}", library_overlay.display())
        };
        environment.insert("LD_LIBRARY_PATH".into(), library_path);

        Ok(Self {
            root,
            config_directory,
            library_overlay,
            runtime_link,
            runtime_target,
            output_directory: plan.paths.output_directory.clone(),
            simulator_dump_directory: plan.paths.simulator_dump_directory.clone(),
            copied_config_files,
            config_patches,
            environment,
            command: plan.command.clone(),
        })
    }
}

#[derive(Debug, Error)]
pub enum WorkspaceError {
    #[error("simulator runtime is missing: {0}")]
    MissingRuntime(PathBuf),
    #[error("no supported model configuration found in {0}")]
    NoModelConfiguration(PathBuf),
    #[error("runtime link already exists with an unexpected target: {path} -> {actual:?}")]
    ConflictingRuntimeLink {
        path: PathBuf,
        actual: Option<PathBuf>,
    },
    #[error("failed to {action} {path}: {source}")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid simulator JSON configuration {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

fn create_dir(path: &Path) -> Result<(), WorkspaceError> {
    fs::create_dir_all(path).map_err(|source| io_error("create directory", path, source))
}

fn create_runtime_link(link: &Path, target: &Path) -> Result<(), WorkspaceError> {
    match fs::symlink_metadata(link) {
        Ok(_) => {
            let actual = fs::read_link(link).ok();
            if actual.as_deref() == Some(target) {
                Ok(())
            } else {
                Err(WorkspaceError::ConflictingRuntimeLink {
                    path: link.to_path_buf(),
                    actual,
                })
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            symlink(target, link).map_err(|source| io_error("create symlink", link, source))
        }
        Err(source) => Err(io_error("inspect", link, source)),
    }
}

fn copy_and_patch_json(
    source: &Path,
    destination: &Path,
    options: WorkspaceOptions,
    patches: &mut Vec<ConfigPatch>,
) -> Result<(), WorkspaceError> {
    let bytes = fs::read(source).map_err(|error| io_error("read", source, error))?;
    let mut document: Value =
        serde_json::from_slice(&bytes).map_err(|error| WorkspaceError::Json {
            path: source.to_path_buf(),
            source: error,
        })?;

    if options.enable_parallel_simulation {
        patch_json_number(&mut document, "/pem/parsim", 1, destination, patches);
        patch_json_number(
            &mut document,
            "/pem/parsim_thd_limit",
            options.parallel_thread_limit,
            destination,
            patches,
        );
    }
    if options.enable_cache {
        patch_json_number(
            &mut document,
            "/L2CACHE/cache_enable",
            1,
            destination,
            patches,
        );
    }
    if let Some(level) = options.flush_level {
        patch_json_number(
            &mut document,
            "/LOG/flush_level",
            level,
            destination,
            patches,
        );
    }

    let mut output =
        serde_json::to_vec_pretty(&document).map_err(|source| WorkspaceError::Json {
            path: destination.to_path_buf(),
            source,
        })?;
    output.push(b'\n');
    fs::write(destination, output).map_err(|error| io_error("write", destination, error))
}

fn copy_and_patch_text(
    source: &Path,
    destination: &Path,
    options: WorkspaceOptions,
    patches: &mut Vec<ConfigPatch>,
) -> Result<(), WorkspaceError> {
    let bytes = fs::read(source).map_err(|error| io_error("read", source, error))?;
    let mut text = String::from_utf8_lossy(&bytes).into_owned();

    if options.enable_parallel_simulation {
        replace_text(&mut text, "parsim = 0", "parsim = 1", destination, patches);
        replace_text(
            &mut text,
            "parsim_thd_limit = 0",
            &format!("parsim_thd_limit = {}", options.parallel_thread_limit),
            destination,
            patches,
        );
    }
    if options.enable_cache {
        replace_text(
            &mut text,
            "cache_enable                = 0",
            "cache_enable                = 1",
            destination,
            patches,
        );
    }
    if let Some(level) = options.flush_level {
        replace_text(
            &mut text,
            "flush_level = \"3\"",
            &format!("flush_level = \"{level}\""),
            destination,
            patches,
        );
    }

    fs::write(destination, text).map_err(|error| io_error("write", destination, error))
}

fn patch_json_number(
    document: &mut Value,
    pointer: &str,
    replacement: u64,
    file: &Path,
    patches: &mut Vec<ConfigPatch>,
) {
    let Some(value) = document.pointer_mut(pointer) else {
        return;
    };
    let before = value.to_string();
    let after = replacement.to_string();
    if before == after {
        return;
    }
    *value = Value::from(replacement);
    patches.push(ConfigPatch {
        file: file.to_path_buf(),
        field: pointer.trim_start_matches('/').replace('/', "."),
        before,
        after,
    });
}

fn replace_text(
    text: &mut String,
    before: &str,
    after: &str,
    file: &Path,
    patches: &mut Vec<ConfigPatch>,
) {
    if !text.contains(before) || before == after {
        return;
    }
    *text = text.replace(before, after);
    patches.push(ConfigPatch {
        file: file.to_path_buf(),
        field: before.to_owned(),
        before: before.to_owned(),
        after: after.to_owned(),
    });
}

fn io_error(action: &'static str, path: impl AsRef<Path>, source: io::Error) -> WorkspaceError {
    WorkspaceError::Io {
        action,
        path: path.as_ref().to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::SimulatorRequest;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("open-ascend-emulator-{}-{id}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn prepares_private_runtime_overlay_and_patched_configs() {
        let tree = TempTree::new();
        let ascend_home = tree.0.join("cann");
        let simulator_lib = ascend_home.join("tools/simulator/dav_2201/lib");
        fs::create_dir_all(&simulator_lib).unwrap();
        fs::write(simulator_lib.join("libruntime_camodel.so"), b"runtime").unwrap();
        fs::write(
            simulator_lib.join("config.json"),
            br#"{"LOG":{"flush_level":3},"L2CACHE":{"cache_enable":0}}"#,
        )
        .unwrap();
        fs::write(
            simulator_lib.join("config_stars.json"),
            br#"{"pem":{"parsim":0,"parsim_thd_limit":0}}"#,
        )
        .unwrap();

        let output = tree.0.join("output");
        let plan = LaunchPlan::build(
            SimulatorRequest {
                output: Some(output),
                soc_version: Some("Ascend910B1".into()),
                ..SimulatorRequest::default()
            },
            &ascend_home,
            Some("/host/lib"),
        )
        .unwrap();
        let prepared =
            PreparedRun::prepare(&plan, tree.0.join("run"), WorkspaceOptions::default()).unwrap();

        assert_eq!(
            fs::read_link(&prepared.runtime_link).unwrap(),
            prepared.runtime_target
        );
        assert_eq!(prepared.copied_config_files.len(), 2);
        assert_eq!(prepared.config_patches.len(), 3);
        assert_eq!(
            prepared.environment["CAMODEL_CONFIG_PATH"],
            prepared.config_directory.display().to_string()
        );
        assert!(
            prepared.environment["LD_LIBRARY_PATH"]
                .starts_with(&prepared.library_overlay.display().to_string())
        );

        let config: Value = serde_json::from_slice(
            &fs::read(prepared.config_directory.join("config.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            config.pointer("/L2CACHE/cache_enable"),
            Some(&Value::from(1))
        );
        let stars: Value = serde_json::from_slice(
            &fs::read(prepared.config_directory.join("config_stars.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(stars.pointer("/pem/parsim"), Some(&Value::from(1)));
        assert_eq!(
            stars.pointer("/pem/parsim_thd_limit"),
            Some(&Value::from(24))
        );
    }

    #[test]
    fn refuses_to_replace_an_existing_runtime_link() {
        let tree = TempTree::new();
        let ascend_home = tree.0.join("cann");
        let simulator_lib = ascend_home.join("tools/simulator/dav_3510/lib");
        fs::create_dir_all(&simulator_lib).unwrap();
        fs::write(simulator_lib.join("libruntime_camodel.so"), b"runtime").unwrap();
        fs::write(simulator_lib.join("config.json"), br#"{}"#).unwrap();
        let run_lib = tree.0.join("run/lib");
        fs::create_dir_all(&run_lib).unwrap();
        symlink("unexpected", run_lib.join("libruntime.so")).unwrap();

        let plan = LaunchPlan::build(
            SimulatorRequest {
                output: Some(tree.0.join("output")),
                soc_version: Some("Ascend950PR_9599".into()),
                ..SimulatorRequest::default()
            },
            ascend_home,
            None,
        )
        .unwrap();
        assert!(matches!(
            PreparedRun::prepare(&plan, tree.0.join("run"), WorkspaceOptions::default()),
            Err(WorkspaceError::ConflictingRuntimeLink { .. })
        ));
    }
}
