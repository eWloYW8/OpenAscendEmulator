use crate::plan::CommandSpec;
use crate::workspace::PreparedRun;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};
use thiserror::Error;

const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(25);
const MAX_RETAINED_DIAGNOSTICS: usize = 256;
const MAX_RETAINED_CONFIG_LOADS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LogStream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticKind {
    AclApiFailure,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VendorDiagnostic {
    pub stream: LogStream,
    pub kind: DiagnosticKind,
    pub line: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelConfigLoad {
    pub stream: LogStream,
    pub file_name: String,
    pub source: String,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactChange {
    Created,
    Modified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArtifactEvidence {
    pub path: PathBuf,
    pub bytes: u64,
    pub change: ArtifactChange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RunRejection {
    TimedOut,
    ProcessStatus,
    VendorDiagnostic,
    ModelStopMissing,
    ArtifactEvidenceMissing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunOutcome {
    pub command: CommandSpec,
    pub run_root: std::path::PathBuf,
    pub elapsed_milliseconds: u128,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub process_success: bool,
    pub success: bool,
    pub timed_out: bool,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
    pub kernel_start_observed: bool,
    pub model_stop_observed: bool,
    pub vendor_diagnostics: Vec<VendorDiagnostic>,
    pub omitted_vendor_diagnostics: usize,
    pub model_config_loads: Vec<ModelConfigLoad>,
    pub omitted_model_config_loads: usize,
    pub changed_artifacts: Vec<ArtifactEvidence>,
    pub rejections: Vec<RunRejection>,
}

impl PreparedRun {
    pub fn execute(&self, timeout: Option<Duration>) -> Result<RunOutcome, RunnerError> {
        let started = Instant::now();
        let logs = self.root.join("logs");
        fs::create_dir_all(&logs).map_err(|source| RunnerError::Io {
            action: "create run log directory",
            source,
        })?;
        let stdout_log = logs.join("application.stdout.log");
        let stderr_log = logs.join("application.stderr.log");
        let stdout_file = File::create(&stdout_log).map_err(|source| RunnerError::Io {
            action: "create stdout log",
            source,
        })?;
        let stderr_file = File::create(&stderr_log).map_err(|source| RunnerError::Io {
            action: "create stderr log",
            source,
        })?;
        let artifacts_before = snapshot_files(&self.output_directory)?;

        let mut child = Command::new(&self.command.executable)
            .args(&self.command.arguments)
            .envs(&self.environment)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| RunnerError::Io {
                action: "spawn workload",
                source,
            })?;
        let stdout = child.stdout.take().ok_or(RunnerError::MissingPipe {
            stream: LogStream::Stdout,
        })?;
        let stderr = child.stderr.take().ok_or(RunnerError::MissingPipe {
            stream: LogStream::Stderr,
        })?;
        let stdout_capture = spawn_capture(stdout, stdout_file, LogStream::Stdout);
        let stderr_capture = spawn_capture(stderr, stderr_file, LogStream::Stderr);

        let deadline = timeout.and_then(|duration| started.checked_add(duration));
        let (status, timed_out) = loop {
            let status = child.try_wait().map_err(|source| RunnerError::Io {
                action: "poll workload",
                source,
            })?;
            if let Some(status) = status {
                break (status, false);
            }

            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                child.kill().map_err(|source| RunnerError::Io {
                    action: "terminate timed-out workload",
                    source,
                })?;
                let status = child.wait().map_err(|source| RunnerError::Io {
                    action: "reap timed-out workload",
                    source,
                })?;
                break (status, true);
            }

            let sleep_for = deadline
                .map(|deadline| deadline.saturating_duration_since(Instant::now()))
                .map(|remaining| remaining.min(WAIT_POLL_INTERVAL))
                .unwrap_or(WAIT_POLL_INTERVAL);
            if !sleep_for.is_zero() {
                thread::sleep(sleep_for);
            }
        };

        let stdout_capture = join_capture(stdout_capture, LogStream::Stdout)?;
        let stderr_capture = join_capture(stderr_capture, LogStream::Stderr)?;
        let artifacts_after = snapshot_files(&self.output_directory)?;
        let changed_artifacts = changed_artifacts(&artifacts_before, &artifacts_after);

        let kernel_start_observed =
            stdout_capture.kernel_start_observed || stderr_capture.kernel_start_observed;
        let model_stop_observed =
            stdout_capture.model_stop_observed || stderr_capture.model_stop_observed;
        let mut vendor_diagnostics = stdout_capture.vendor_diagnostics;
        vendor_diagnostics.extend(stderr_capture.vendor_diagnostics);
        let omitted_vendor_diagnostics =
            stdout_capture.omitted_vendor_diagnostics + stderr_capture.omitted_vendor_diagnostics;
        let mut model_config_loads = stdout_capture.model_config_loads;
        model_config_loads.extend(stderr_capture.model_config_loads);
        let omitted_model_config_loads =
            stdout_capture.omitted_model_config_loads + stderr_capture.omitted_model_config_loads;
        let process_success = status.success() && !timed_out;

        let mut rejections = Vec::new();
        if timed_out {
            rejections.push(RunRejection::TimedOut);
        } else if !process_success {
            rejections.push(RunRejection::ProcessStatus);
        }
        if !vendor_diagnostics.is_empty() || omitted_vendor_diagnostics != 0 {
            rejections.push(RunRejection::VendorDiagnostic);
        }
        if kernel_start_observed && !model_stop_observed {
            rejections.push(RunRejection::ModelStopMissing);
        }
        if kernel_start_observed && changed_artifacts.is_empty() {
            rejections.push(RunRejection::ArtifactEvidenceMissing);
        }

        Ok(RunOutcome {
            command: self.command.clone(),
            run_root: self.root.clone(),
            elapsed_milliseconds: started.elapsed().as_millis(),
            exit_code: status.code(),
            signal: status.signal(),
            process_success,
            success: rejections.is_empty(),
            timed_out,
            stdout_log,
            stderr_log,
            kernel_start_observed,
            model_stop_observed,
            vendor_diagnostics,
            omitted_vendor_diagnostics,
            model_config_loads,
            omitted_model_config_loads,
            changed_artifacts,
            rejections,
        })
    }
}

#[derive(Debug, Default)]
struct StreamCapture {
    kernel_start_observed: bool,
    model_stop_observed: bool,
    vendor_diagnostics: Vec<VendorDiagnostic>,
    omitted_vendor_diagnostics: usize,
    model_config_loads: Vec<ModelConfigLoad>,
    omitted_model_config_loads: usize,
}

fn spawn_capture<R>(
    reader: R,
    log: File,
    stream: LogStream,
) -> thread::JoinHandle<Result<StreamCapture, io::Error>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || capture_stream(reader, log, stream))
}

fn capture_stream<R: Read>(
    reader: R,
    mut log: File,
    stream: LogStream,
) -> Result<StreamCapture, io::Error> {
    let mut reader = BufReader::new(reader);
    let mut capture = StreamCapture::default();
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            break;
        }
        log.write_all(&line)?;
        write_terminal(stream, &line);
        inspect_line(&mut capture, stream, &line);
    }
    log.flush()?;
    Ok(capture)
}

fn write_terminal(stream: LogStream, bytes: &[u8]) {
    let result = match stream {
        LogStream::Stdout => io::stdout().lock().write_all(bytes),
        LogStream::Stderr => io::stderr().lock().write_all(bytes),
    };
    let _ = result;
}

fn inspect_line(capture: &mut StreamCapture, stream: LogStream, bytes: &[u8]) {
    let line = String::from_utf8_lossy(bytes);
    if let Some(load) = parse_model_config_load(&line, stream) {
        if capture.model_config_loads.len() < MAX_RETAINED_CONFIG_LOADS {
            capture.model_config_loads.push(load);
        } else {
            capture.omitted_model_config_loads += 1;
        }
    }
    capture.kernel_start_observed |= line.contains("<ProfInit> Start profiling on kernel:");
    capture.model_stop_observed |= line.contains("[INFO] Model stopped successfully.");
    if line.contains("[ERROR] <CheckResult>") && line.contains("failed") {
        let diagnostic = VendorDiagnostic {
            stream,
            kind: DiagnosticKind::AclApiFailure,
            line: line.trim_end_matches(['\r', '\n']).to_owned(),
        };
        if capture.vendor_diagnostics.len() < MAX_RETAINED_DIAGNOSTICS {
            capture.vendor_diagnostics.push(diagnostic);
        } else {
            capture.omitted_vendor_diagnostics += 1;
        }
    }
}

fn parse_model_config_load(line: &str, stream: LogStream) -> Option<ModelConfigLoad> {
    let line = line.trim_end_matches(['\r', '\n']);
    let body = line.strip_prefix("[INFO] Config file [")?;
    let (file_name, rest) = body.split_once("] from ")?;
    let (source, path) = rest.split_once(". Path: ")?;
    if file_name.is_empty() || source.is_empty() || path.is_empty() {
        return None;
    }
    Some(ModelConfigLoad {
        stream,
        file_name: file_name.to_owned(),
        source: source.to_owned(),
        path: PathBuf::from(path),
    })
}

fn join_capture(
    handle: thread::JoinHandle<Result<StreamCapture, io::Error>>,
    stream: LogStream,
) -> Result<StreamCapture, RunnerError> {
    handle
        .join()
        .map_err(|_| RunnerError::CaptureThreadPanicked { stream })?
        .map_err(|source| RunnerError::Io {
            action: "capture workload output",
            source,
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    bytes: u64,
    modified_nanoseconds: Option<u128>,
}

fn snapshot_files(root: &Path) -> Result<BTreeMap<PathBuf, FileStamp>, RunnerError> {
    let mut files = BTreeMap::new();
    snapshot_directory(root, &mut files)?;
    Ok(files)
}

fn snapshot_directory(
    directory: &Path,
    files: &mut BTreeMap<PathBuf, FileStamp>,
) -> Result<(), RunnerError> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(source) => {
            return Err(RunnerError::Io {
                action: "scan simulator output directory",
                source,
            });
        }
    };
    for entry in entries {
        let entry = entry.map_err(|source| RunnerError::Io {
            action: "read simulator output entry",
            source,
        })?;
        let path = entry.path();
        let metadata = entry.metadata().map_err(|source| RunnerError::Io {
            action: "inspect simulator output entry",
            source,
        })?;
        if metadata.is_dir() {
            snapshot_directory(&path, files)?;
        } else if metadata.is_file() {
            let modified_nanoseconds = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_nanos());
            files.insert(
                path,
                FileStamp {
                    bytes: metadata.len(),
                    modified_nanoseconds,
                },
            );
        }
    }
    Ok(())
}

fn changed_artifacts(
    before: &BTreeMap<PathBuf, FileStamp>,
    after: &BTreeMap<PathBuf, FileStamp>,
) -> Vec<ArtifactEvidence> {
    after
        .iter()
        .filter_map(|(path, stamp)| match before.get(path) {
            None => Some(ArtifactEvidence {
                path: path.clone(),
                bytes: stamp.bytes,
                change: ArtifactChange::Created,
            }),
            Some(previous) if previous != stamp => Some(ArtifactEvidence {
                path: path.clone(),
                bytes: stamp.bytes,
                change: ArtifactChange::Modified,
            }),
            Some(_) => None,
        })
        .collect()
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("failed to {action}: {source}")]
    Io {
        action: &'static str,
        #[source]
        source: io::Error,
    },
    #[error("workload {stream:?} pipe was not available")]
    MissingPipe { stream: LogStream },
    #[error("workload {stream:?} capture thread panicked")]
    CaptureThreadPanicked { stream: LogStream },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let id = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "open-ascend-emulator-runner-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn prepared(
        tree: &TempTree,
        command: CommandSpec,
        environment: BTreeMap<String, String>,
    ) -> PreparedRun {
        let output_directory = tree.0.join("output");
        fs::create_dir_all(&output_directory).unwrap();
        PreparedRun {
            root: tree.0.join("run"),
            config_directory: PathBuf::new(),
            library_overlay: PathBuf::new(),
            runtime_link: PathBuf::new(),
            runtime_target: PathBuf::new(),
            simulator_dump_directory: output_directory.join("device0/tmp_dump"),
            output_directory,
            copied_config_files: Vec::new(),
            config_patches: Vec::new(),
            environment,
            command,
        }
    }

    #[test]
    fn executes_without_a_shell_and_overlays_environment() {
        let tree = TempTree::new();
        let mut environment = BTreeMap::new();
        environment.insert("OPEN_ASCEND_RUNNER_TEST".into(), "expected".into());
        let run = prepared(
            &tree,
            CommandSpec {
                executable: PathBuf::from("/bin/sh"),
                arguments: vec![
                    "-c".into(),
                    "test \"$OPEN_ASCEND_RUNNER_TEST\" = expected".into(),
                ],
            },
            environment,
        );

        let outcome = run.execute(Some(Duration::from_secs(2))).unwrap();
        assert!(outcome.success);
        assert!(outcome.process_success);
        assert!(!outcome.timed_out);
        assert_eq!(outcome.exit_code, Some(0));
        assert!(outcome.stdout_log.is_file());
        assert!(outcome.stderr_log.is_file());
    }

    #[test]
    fn terminates_and_reaps_a_timed_out_process() {
        let tree = TempTree::new();
        let run = prepared(
            &tree,
            CommandSpec {
                executable: PathBuf::from("/bin/sleep"),
                arguments: vec!["5".into()],
            },
            BTreeMap::new(),
        );

        let outcome = run.execute(Some(Duration::from_millis(30))).unwrap();
        assert!(outcome.timed_out);
        assert!(!outcome.success);
        assert_eq!(outcome.signal, Some(9));
        assert!(outcome.elapsed_milliseconds < 1000);
        assert_eq!(outcome.rejections, vec![RunRejection::TimedOut]);
    }

    #[test]
    fn rejects_vendor_acl_failure_even_when_process_exits_zero() {
        let tree = TempTree::new();
        let run = prepared(
            &tree,
            CommandSpec {
                executable: PathBuf::from("/bin/sh"),
                arguments: vec![
                    "-c".into(),
                    "printf '%s\\n' '[ERROR] <CheckResult> Aclrt API call aclrtKernelArgsAppend() failed. error code: 100000'".into(),
                ],
            },
            BTreeMap::new(),
        );

        let outcome = run.execute(Some(Duration::from_secs(2))).unwrap();
        assert!(outcome.process_success);
        assert!(!outcome.success);
        assert_eq!(outcome.vendor_diagnostics.len(), 1);
        assert_eq!(
            outcome.vendor_diagnostics[0].kind,
            DiagnosticKind::AclApiFailure
        );
        assert_eq!(outcome.rejections, vec![RunRejection::VendorDiagnostic]);
    }

    #[test]
    fn accepts_completed_kernel_lifecycle_with_new_artifact() {
        let tree = TempTree::new();
        let output = tree.0.join("output");
        let run = prepared(
            &tree,
            CommandSpec {
                executable: PathBuf::from("/bin/sh"),
                arguments: vec![
                    "-c".into(),
                    "printf '%s\\n' '[INFO] <ProfInit> Start profiling on kernel: EmptyKernel'; printf '%s\\n' '[ERROR] <SendMsg> Send one DBI data failed, error ret=0'; printf artifact > \"$1/aicore_binary.o\"; printf '%s\\n' '[INFO] Model stopped successfully.'".into(),
                    "runner-test".into(),
                    output.display().to_string(),
                ],
            },
            BTreeMap::new(),
        );

        let outcome = run.execute(Some(Duration::from_secs(2))).unwrap();
        assert!(outcome.success);
        assert!(outcome.kernel_start_observed);
        assert!(outcome.model_stop_observed);
        assert!(outcome.vendor_diagnostics.is_empty());
        assert_eq!(outcome.changed_artifacts.len(), 1);
        assert_eq!(outcome.changed_artifacts[0].bytes, 8);
        assert_eq!(outcome.changed_artifacts[0].change, ArtifactChange::Created);
    }

    #[test]
    fn rejects_incomplete_kernel_lifecycle_without_artifacts() {
        let tree = TempTree::new();
        let run = prepared(
            &tree,
            CommandSpec {
                executable: PathBuf::from("/bin/sh"),
                arguments: vec![
                    "-c".into(),
                    "printf '%s\\n' '[INFO] <ProfInit> Start profiling on kernel: EmptyKernel'"
                        .into(),
                ],
            },
            BTreeMap::new(),
        );

        let outcome = run.execute(Some(Duration::from_secs(2))).unwrap();
        assert!(outcome.process_success);
        assert!(!outcome.success);
        assert_eq!(
            outcome.rejections,
            vec![
                RunRejection::ModelStopMissing,
                RunRejection::ArtifactEvidenceMissing
            ]
        );
    }

    #[test]
    fn retains_reported_model_config_provenance_without_assuming_private_copy() {
        let mut capture = StreamCapture::default();
        inspect_line(
            &mut capture,
            LogStream::Stdout,
            b"[INFO] Config file [config_stars.json] from environment variable [CAMODEL_CONFIG_PATH]. Path: /run/private/config/config_stars.json\n",
        );
        inspect_line(
            &mut capture,
            LogStream::Stdout,
            b"[INFO] Config file [config.json] from model dynamic library path. Path: /usr/local/Ascend/cann-9.0.1/aarch64-linux/simulator/dav_3510/lib/config.json\n",
        );
        inspect_line(
            &mut capture,
            LogStream::Stdout,
            b"[INFO] Config file is found, path is /run/private/config/config_stars.json.\n",
        );
        assert_eq!(capture.model_config_loads.len(), 2);
        assert_eq!(capture.model_config_loads[0].file_name, "config_stars.json");
        assert_eq!(
            capture.model_config_loads[0].source,
            "environment variable [CAMODEL_CONFIG_PATH]"
        );
        assert_eq!(
            capture.model_config_loads[1].source,
            "model dynamic library path"
        );
        assert_eq!(
            capture.model_config_loads[1].path,
            PathBuf::from(
                "/usr/local/Ascend/cann-9.0.1/aarch64-linux/simulator/dav_3510/lib/config.json"
            )
        );

        for _ in 0..MAX_RETAINED_CONFIG_LOADS {
            inspect_line(
                &mut capture,
                LogStream::Stderr,
                b"[INFO] Config file [x.json] from model dynamic library path. Path: /x.json\n",
            );
        }
        assert_eq!(capture.model_config_loads.len(), MAX_RETAINED_CONFIG_LOADS);
        assert_eq!(capture.omitted_model_config_loads, 2);
    }
}
