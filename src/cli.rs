use crate::acl_args::{AclArgumentPlan, AclArgumentPlanError};
use crate::architecture::{Architecture, all_device_profiles};
use crate::device_elf::{
    DeviceElf, DeviceElfError, DeviceElfHeader, DeviceKernelSummary, DeviceLoadImageSummary,
};
use crate::isa::{AicDecoderHint, AicFramingError, AicInstructionWord, AicWordFramer};
use crate::kernel_config::{KernelConfigDocument, KernelConfigError};
use crate::plan::{LaunchPlan, PlanError, SimulatorRequest};
use crate::replay_seed::{ReplaySeed, ReplaySeedError};
use crate::trace::verify_scalar_trace;
use crate::workspace::{PreparedRun, WorkspaceError, WorkspaceOptions};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::Serialize;
use std::env;
use std::fs::{self, File};
use std::io;
use std::io::BufReader;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

fn parse_u64_address(raw: &str) -> Result<u64, String> {
    if let Some(digits) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(digits, 16).map_err(|_| format!("invalid hexadecimal address: {raw}"))
    } else {
        raw.parse()
            .map_err(|_| format!("invalid decimal address: {raw}"))
    }
}

fn parse_pc_start_addr(bytes: &[u8]) -> Result<u64, &'static str> {
    let raw = std::str::from_utf8(bytes).map_err(|_| "not UTF-8")?;
    let raw = raw
        .strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(raw);
    let digits = raw.strip_prefix("0x").ok_or("expected 0x-prefixed hex")?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("expected hex digits with no surrounding text");
    }
    let address = u64::from_str_radix(digits, 16).map_err(|_| "address exceeds 64 bits")?;
    if address == 0 {
        return Err("PC start address is zero");
    }
    Ok(address)
}

fn read_pc_start_addr(path: &Path) -> Result<u64, CliError> {
    let file = File::open(path).map_err(|source| CliError::FileRead {
        path: path.to_path_buf(),
        source,
    })?;
    let mut bytes = Vec::new();
    file.take(65)
        .read_to_end(&mut bytes)
        .map_err(|source| CliError::FileRead {
            path: path.to_path_buf(),
            source,
        })?;
    if bytes.len() > 64 {
        return Err(CliError::InvalidPcStartAddrFile {
            path: path.to_path_buf(),
            detail: "file exceeds 64 bytes",
        });
    }
    parse_pc_start_addr(&bytes).map_err(|detail| CliError::InvalidPcStartAddrFile {
        path: path.to_path_buf(),
        detail,
    })
}

#[derive(Debug, Parser)]
#[command(name = "open-ascend-emulator", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Op {
        #[command(subcommand)]
        command: OpCommand,
    },
    Devices {
        #[arg(long)]
        json: bool,
    },
    InspectConfig {
        path: PathBuf,
        #[arg(long, conflicts_with = "stage_inputs")]
        acl_args: bool,
        #[arg(long, requires = "max_loaded_bytes")]
        stage_inputs: bool,
        #[arg(long, requires = "stage_inputs")]
        max_loaded_bytes: Option<u64>,
        #[arg(long)]
        json: bool,
    },
    InspectElf {
        path: PathBuf,
        #[arg(long)]
        kernel: Option<String>,
        #[arg(long, requires = "kernel")]
        aic_architecture: Option<Architecture>,
        #[arg(long, default_value_t = 32)]
        max_words: usize,
        #[arg(long, requires = "kernel", conflicts_with = "pc_start_addr_file", value_parser = parse_u64_address)]
        aligned_code_base: Option<u64>,
        #[arg(long, requires = "kernel", conflicts_with = "aligned_code_base")]
        pc_start_addr_file: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    VerifyScalarTrace {
        path: PathBuf,
        #[arg(long)]
        architecture: Architecture,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum OpCommand {
    Simulator(Box<SimulatorArgs>),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum OnOff {
    On,
    Off,
}

impl From<OnOff> for bool {
    fn from(value: OnOff) -> Self {
        matches!(value, OnOff::On)
    }
}

#[derive(Debug, Args)]
struct SimulatorArgs {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    application: Option<PathBuf>,
    #[arg(long)]
    export: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    kernel_name: Option<String>,
    #[arg(long, value_delimiter = ',')]
    aic_metrics: Vec<String>,
    #[arg(long)]
    launch_count: Option<u32>,
    #[arg(long)]
    mstx: Option<OnOff>,
    #[arg(long)]
    mstx_include: Option<String>,
    #[arg(long)]
    soc_version: Option<String>,
    #[arg(long, value_delimiter = ',')]
    core_id: Vec<u32>,
    #[arg(long = "timeout")]
    timeout_minutes: Option<u32>,
    #[arg(long)]
    dump: Option<OnOff>,
    #[arg(long)]
    ascend_home: Option<PathBuf>,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    prepare_dir: Option<PathBuf>,
    #[arg(long, requires = "prepare_dir", requires = "application")]
    execute: bool,
    #[arg(last = true)]
    application_args: Vec<String>,
}

#[derive(Debug, Error)]
pub enum CliError {
    #[error("ASCEND_HOME_PATH is unset; pass --ascend-home")]
    MissingAscendHome,
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("failed to serialize output: {0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    KernelConfig(#[from] KernelConfigError),
    #[error(transparent)]
    AclArguments(#[from] AclArgumentPlanError),
    #[error(transparent)]
    ReplaySeed(#[from] ReplaySeedError),
    #[error("failed to read {path}: {source}")]
    FileRead {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid simulator PC-start file {path}: {detail}")]
    InvalidPcStartAddrFile { path: PathBuf, detail: &'static str },
    #[error(transparent)]
    DeviceElf(#[from] DeviceElfError),
    #[error(transparent)]
    AicFraming(#[from] AicFramingError),
    #[error("failed to read scalar trace: {0}")]
    TraceRead(#[from] io::Error),
    #[error("scalar trace contained no supported scalar arithmetic, move, or ZEROEXT records")]
    EmptyScalarTrace,
    #[error("scalar trace has {0} mismatches")]
    ScalarTraceMismatch(u64),
    #[error(transparent)]
    Workspace(#[from] WorkspaceError),
    #[error(transparent)]
    Runner(#[from] crate::runner::RunnerError),
    #[error("simulator workload timed out")]
    WorkloadTimedOut,
    #[error("simulator workload failed with exit code {exit_code:?} or signal {signal:?}")]
    WorkloadFailed {
        exit_code: Option<i32>,
        signal: Option<i32>,
    },
    #[error("simulator workload failed verification: {reasons}")]
    WorkloadRejected { reasons: String },
}

pub fn run() -> Result<(), CliError> {
    run_with(Cli::parse())
}

fn run_with(cli: Cli) -> Result<(), CliError> {
    match cli.command {
        Command::Devices { json } => {
            let devices: Vec<_> = all_device_profiles().collect();
            if json {
                print_json(&devices)?;
            } else {
                for device in devices {
                    println!(
                        "{}\t{}\tAIC={}\tAIV={}",
                        device.soc_version,
                        device.architecture,
                        device.simulator_aic_cores,
                        device.simulator_aiv_cores
                    );
                }
            }
        }
        Command::InspectConfig {
            path,
            acl_args,
            stage_inputs,
            max_loaded_bytes,
            json,
        } => {
            let document = KernelConfigDocument::from_path(path)?;
            let config = document.decode()?;
            if stage_inputs {
                let seed = ReplaySeed::load(
                    &config,
                    max_loaded_bytes.expect("clap requires a staging byte limit"),
                )?;
                let summary = seed.summary();
                if json {
                    print_json(&summary)?;
                } else {
                    println!("pointer-argument-bytes: {}", summary.append_bytes);
                    println!("arguments: {}", summary.arguments.len());
                    for region in summary.regions {
                        println!(
                            "region[{}]: {:?} allocation={} known-prefix={} unknown-tail={}",
                            region.index,
                            region.kind,
                            region.allocation_bytes,
                            region.known_prefix_bytes,
                            region.unknown_tail_bytes
                        );
                    }
                }
            } else if acl_args {
                let plan = AclArgumentPlan::from_config(&config)?;
                if json {
                    print_json(&plan)?;
                } else {
                    println!("pointer-argument-bytes: {}", plan.append_bytes);
                    for slot in plan.slots {
                        println!(
                            "arg[{}]: {:?} path={:?} logical={:?} host-alloc={:?} device-alloc={:?} h2d={:?} d2h={:?} null={}",
                            slot.index,
                            slot.kind,
                            slot.path,
                            slot.logical_bytes,
                            slot.host_allocation_bytes,
                            slot.device_allocation_bytes,
                            slot.host_to_device_copy_bytes,
                            slot.device_to_host_copy_bytes,
                            slot.passes_null_pointer
                        );
                    }
                }
            } else if json {
                print_json(&config)?;
            } else {
                println!("runner: {:?}", config.runner);
                println!("kernel: {}", config.kernel_name);
                println!("binary: {}", config.bin_path);
                println!("block-dim: {}", config.block_dim);
                println!("device-id: {}", config.device_id);
                println!("inputs: {}", config.inputs.len());
                println!("outputs: {}", config.outputs.len());
                println!("fields: {}", config.present_fields.join(","));
                if !config.unknown_fields.is_empty() {
                    println!(
                        "unknown-fields: {}",
                        config
                            .unknown_fields
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
            }
        }
        Command::InspectElf {
            path,
            kernel,
            aic_architecture,
            max_words,
            aligned_code_base,
            pc_start_addr_file,
            json,
        } => {
            let bytes = fs::read(&path).map_err(|source| CliError::FileRead {
                path: path.clone(),
                source,
            })?;
            let elf = DeviceElf::parse(&bytes)?;
            let (load_image, load_image_error) = match elf.load_image() {
                Ok(image) => (Some(image.summary), None),
                Err(error) => (None, Some(error.to_string())),
            };
            let kernels = elf.kernels()?;
            let code_base = if let Some(path) = pc_start_addr_file {
                Some(CodeBaseEvidence::PcStartAddrFile {
                    address: read_pc_start_addr(&path)?,
                    path,
                })
            } else {
                aligned_code_base.map(|address| CodeBaseEvidence::SuppliedArgument { address })
            };
            let selected = if let Some(name) = kernel {
                let image = elf.kernel(&name)?;
                let projected = code_base
                    .as_ref()
                    .map(|base| elf.project_kernel(&name, base.address()))
                    .transpose()?;
                let device_entry_address = projected.as_ref().map(|item| item.entry_address);
                let mut raw_words = Vec::new();
                let mut aic_frames = aic_architecture.map(|_| Vec::new());
                let mut framer = aic_architecture.map(AicWordFramer::new);
                for (pc, word) in image.words().take(max_words) {
                    let (device_pc, word) = if let Some(projected) = projected.as_ref() {
                        let symbol_offset = pc
                            .checked_sub(image.summary.virtual_address)
                            .ok_or(DeviceElfError::RangeOverflow)?;
                        let device_pc = projected
                            .entry_address
                            .checked_add(symbol_offset)
                            .ok_or(DeviceElfError::RuntimeAddressOverflow)?;
                        (Some(device_pc), projected.fetch_word(device_pc)?)
                    } else {
                        (None, word)
                    };
                    raw_words.push(RawWord {
                        pc,
                        device_pc,
                        word,
                        decoder_hint: aic_architecture
                            .and_then(|architecture| AicDecoderHint::from_word(architecture, word)),
                    });
                    if let Some(framer) = framer.as_mut()
                        && let Some(decoded) = framer.push(pc, word)?
                    {
                        aic_frames
                            .as_mut()
                            .expect("AIC frames enabled")
                            .push(decoded);
                    }
                }
                let preview_complete = raw_words.len() * 4 == image.bytes.len();
                let pending_at_preview_end = framer
                    .as_ref()
                    .is_some_and(|framer| framer.finish().is_err());
                if preview_complete && let Some(framer) = framer.as_ref() {
                    framer.finish()?;
                }
                Some(KernelPreview {
                    summary: image.summary,
                    device_entry_address,
                    word_count: image.bytes.len() / 4,
                    raw_words,
                    aic_frames,
                    preview_complete,
                    pending_at_preview_end,
                })
            } else {
                None
            };
            let inspection = ElfInspection {
                header: elf.header(),
                load_image,
                load_image_error,
                code_base,
                kernels,
                selected,
            };
            if json {
                print_json(&inspection)?;
            } else {
                println!("machine: {:#x}", inspection.header.machine);
                println!("file-type: {}", inspection.header.file_type);
                println!("flags: {:#x}", inspection.header.flags);
                println!("sections: {}", inspection.header.section_count);
                if let Some(load_image) = &inspection.load_image {
                    println!(
                        "load-image: offset={:#x} bytes={} first={} last={} allocated-sections={}",
                        load_image.file_offset,
                        load_image.byte_count,
                        load_image.first_alloc_section,
                        load_image.last_alloc_section,
                        load_image.alloc_section_count
                    );
                } else if let Some(error) = &inspection.load_image_error {
                    println!("load-image: unavailable ({error})");
                }
                if let Some(base) = &inspection.code_base {
                    match base {
                        CodeBaseEvidence::SuppliedArgument { address } => {
                            println!("code-base: {address:#x} (supplied argument)");
                        }
                        CodeBaseEvidence::PcStartAddrFile { address, path } => {
                            println!("code-base: {address:#x} (pc-start file {})", path.display());
                        }
                    }
                }
                for kernel in &inspection.kernels {
                    println!(
                        "kernel: {} section={} address={:#x} bytes={}",
                        kernel.name, kernel.section, kernel.virtual_address, kernel.byte_count
                    );
                }
                if let Some(preview) = inspection.selected {
                    println!("selected-kernel: {}", preview.summary.name);
                    if let Some(address) = preview.device_entry_address {
                        println!("derived-device-entry: {address:#x}");
                    }
                    println!("word-count: {}", preview.word_count);
                    for word in &preview.raw_words {
                        if let Some(device_pc) = word.device_pc {
                            print!(
                                "  {:#x} [device {device_pc:#x}]: {:#010x}",
                                word.pc, word.word
                            );
                        } else {
                            print!("  {:#x}: {:#010x}", word.pc, word.word);
                        }
                        if let Some(hint) = word.decoder_hint {
                            println!(" {hint:?}");
                        } else {
                            println!();
                        }
                    }
                    if let Some(frames) = preview.aic_frames {
                        for frame in frames {
                            println!("  aic-frame: {frame:?}");
                        }
                        println!("pending-at-preview-end: {}", preview.pending_at_preview_end);
                    }
                }
            }
        }
        Command::VerifyScalarTrace {
            path,
            architecture,
            json,
        } => {
            let summary = if path.as_os_str() == "-" {
                verify_scalar_trace(io::stdin().lock(), architecture)?
            } else {
                let file = File::open(&path).map_err(|source| CliError::FileRead {
                    path: path.clone(),
                    source,
                })?;
                verify_scalar_trace(BufReader::new(file), architecture)?
            };
            if json {
                print_json(&summary)?;
            } else {
                println!("architecture: {}", summary.architecture);
                println!("lines: {}", summary.total_lines);
                println!("checked-s64: {}", summary.checked_s64);
                println!("add-immediate: {}", summary.add_immediate);
                println!("multiply-immediate: {}", summary.multiply_immediate);
                println!("subtract-immediate: {}", summary.subtract_immediate);
                println!("add-register: {}", summary.add_register);
                println!("multiply-register: {}", summary.multiply_register);
                println!("and-register: {}", summary.and_register);
                println!("or-register: {}", summary.or_register);
                println!("shift-left-fields: {}", summary.shift_left_fields);
                println!(
                    "shift-left-value-checked: {}",
                    summary.shift_left_value_checked
                );
                println!(
                    "shift-left-unknown-prior: {}",
                    summary.shift_left_unknown_prior
                );
                println!("move-immediate: {}", summary.checked_move_immediate);
                println!("move-keep-lane: {}", summary.checked_move_keep_lane);
                println!("move-register: {}", summary.checked_register_move);
                println!("zero-extend-u8: {}", summary.zero_extend_u8);
                println!("zero-extend-u16: {}", summary.zero_extend_u16);
                println!("zero-extend-u32: {}", summary.zero_extend_u32);
                println!("unsupported-dtype: {}", summary.skipped_unsupported_dtype);
                println!("mismatches: {}", summary.mismatches);
                for mismatch in &summary.mismatch_examples {
                    println!("  line {}: {}", mismatch.line_number, mismatch.reason);
                }
            }
            if summary.checked_s64
                + summary.add_register
                + summary.multiply_register
                + summary.and_register
                + summary.or_register
                + summary.shift_left_fields
                + summary.checked_move_immediate
                + summary.checked_move_keep_lane
                + summary.checked_register_move
                + summary.zero_extend_u8
                + summary.zero_extend_u16
                + summary.zero_extend_u32
                == 0
            {
                return Err(CliError::EmptyScalarTrace);
            }
            if summary.mismatches != 0 {
                return Err(CliError::ScalarTraceMismatch(summary.mismatches));
            }
        }
        Command::Op {
            command: OpCommand::Simulator(arguments),
        } => {
            let prepare_dir = arguments.prepare_dir.clone();
            let output = arguments
                .output
                .or_else(|| prepare_dir.as_ref().map(|root| root.join("output")));
            let ascend_home = arguments
                .ascend_home
                .or_else(|| env::var_os("ASCEND_HOME_PATH").map(PathBuf::from))
                .ok_or(CliError::MissingAscendHome)?;
            let request = SimulatorRequest {
                config: arguments.config,
                application: arguments.application,
                export: arguments.export,
                output,
                kernel_name: arguments.kernel_name,
                aic_metrics: arguments.aic_metrics,
                launch_count: arguments.launch_count,
                mstx: arguments.mstx.map(Into::into),
                mstx_include: arguments.mstx_include,
                soc_version: arguments.soc_version,
                core_ids: arguments.core_id,
                timeout_minutes: arguments.timeout_minutes,
                dump: arguments.dump.map(Into::into),
                application_args: arguments.application_args,
            };
            let inherited_ld_path = env::var("LD_LIBRARY_PATH").ok();
            let plan = LaunchPlan::build(request, ascend_home, inherited_ld_path.as_deref())?;
            if let Some(prepare_dir) = prepare_dir {
                let prepared =
                    PreparedRun::prepare(&plan, prepare_dir, WorkspaceOptions::default())?;
                if arguments.execute {
                    let timeout = arguments
                        .timeout_minutes
                        .map(|minutes| Duration::from_secs(u64::from(minutes) * 60));
                    let outcome = prepared.execute(timeout)?;
                    if arguments.json {
                        print_json(&outcome)?;
                    } else {
                        println!("elapsed-ms: {}", outcome.elapsed_milliseconds);
                        println!("exit-code: {:?}", outcome.exit_code);
                        println!("signal: {:?}", outcome.signal);
                        println!("process-success: {}", outcome.process_success);
                        println!("simulator-success: {}", outcome.success);
                        println!("stdout-log: {}", outcome.stdout_log.display());
                        println!("stderr-log: {}", outcome.stderr_log.display());
                        println!("vendor-diagnostics: {}", outcome.vendor_diagnostics.len());
                        println!("changed-artifacts: {}", outcome.changed_artifacts.len());
                    }
                    if outcome.timed_out {
                        return Err(CliError::WorkloadTimedOut);
                    }
                    if !outcome.process_success {
                        return Err(CliError::WorkloadFailed {
                            exit_code: outcome.exit_code,
                            signal: outcome.signal,
                        });
                    }
                    if !outcome.success {
                        return Err(CliError::WorkloadRejected {
                            reasons: outcome
                                .rejections
                                .iter()
                                .map(|reason| format!("{reason:?}"))
                                .collect::<Vec<_>>()
                                .join(", "),
                        });
                    }
                } else if arguments.json {
                    print_json(&prepared)?;
                } else {
                    println!("run-root: {}", prepared.root.display());
                    println!("config: {}", prepared.config_directory.display());
                    println!("runtime: {}", prepared.runtime_link.display());
                    println!("patches: {}", prepared.config_patches.len());
                }
            } else if arguments.json {
                print_json(&plan)?;
            } else {
                println!("target: {}", plan.target.camodel_soc_version);
                println!("architecture: {}", plan.target.architecture);
                println!("runner: {}", plan.command.executable.display());
                println!(
                    "simulator: {}",
                    plan.paths.simulator_library_directory.display()
                );
                println!("unresolved: {}", plan.unresolved.join("; "));
            }
        }
    }
    Ok(())
}

fn print_json(value: &impl Serialize) -> Result<(), serde_json::Error> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[derive(Debug, Serialize)]
struct RawWord {
    pc: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_pc: Option<u64>,
    word: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoder_hint: Option<AicDecoderHint>,
}

#[derive(Debug, Serialize)]
struct KernelPreview {
    summary: DeviceKernelSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    device_entry_address: Option<u64>,
    word_count: usize,
    raw_words: Vec<RawWord>,
    aic_frames: Option<Vec<AicInstructionWord>>,
    preview_complete: bool,
    pending_at_preview_end: bool,
}

#[derive(Debug, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
enum CodeBaseEvidence {
    SuppliedArgument { address: u64 },
    PcStartAddrFile { address: u64, path: PathBuf },
}

impl CodeBaseEvidence {
    const fn address(&self) -> u64 {
        match self {
            Self::SuppliedArgument { address } | Self::PcStartAddrFile { address, .. } => *address,
        }
    }
}

#[derive(Debug, Serialize)]
struct ElfInspection {
    header: DeviceElfHeader,
    load_image: Option<DeviceLoadImageSummary>,
    load_image_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code_base: Option<CodeBaseEvidence>,
    kernels: Vec<DeviceKernelSummary>,
    selected: Option<KernelPreview>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parses_reference_simulator_options() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "op",
            "simulator",
            "--ascend-home",
            "/opt/ascend",
            "--application",
            "./add",
            "--soc-version",
            "Ascend950DT_9588",
            "--aic-metrics",
            "PipeUtilization,ResourceConflictRatio",
            "--launch-count",
            "3",
            "--core-id",
            "0,3",
            "--timeout",
            "10",
            "--dump",
            "on",
            "--prepare-dir",
            "/tmp/open-ascend-run",
            "--execute",
            "--",
            "input.bin",
        ])
        .unwrap();

        let Command::Op {
            command: OpCommand::Simulator(arguments),
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(arguments.core_id, vec![0, 3]);
        assert_eq!(arguments.application_args, vec!["input.bin"]);
        assert_eq!(arguments.timeout_minutes, Some(10));
        assert_eq!(
            arguments.prepare_dir,
            Some(PathBuf::from("/tmp/open-ascend-run"))
        );
        assert!(arguments.execute);
    }

    #[test]
    fn parses_device_elf_inspection_with_explicit_aic_front_end() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-elf",
            "kernel.o",
            "--kernel",
            "ClearL2Cache",
            "--aic-architecture",
            "dav_3510",
            "--max-words",
            "55",
            "--aligned-code-base",
            "0x10001000",
            "--json",
        ])
        .unwrap();
        let Command::InspectElf {
            path,
            kernel,
            aic_architecture,
            max_words,
            aligned_code_base,
            pc_start_addr_file,
            json,
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(path, PathBuf::from("kernel.o"));
        assert_eq!(kernel.as_deref(), Some("ClearL2Cache"));
        assert_eq!(aic_architecture, Some(Architecture::Dav3510));
        assert_eq!(max_words, 55);
        assert_eq!(aligned_code_base, Some(0x1000_1000));
        assert!(pc_start_addr_file.is_none());
        assert!(json);

        assert!(
            Cli::try_parse_from([
                "open-ascend-emulator",
                "inspect-elf",
                "kernel.o",
                "--aic-architecture",
                "dav_2201"
            ])
            .is_err()
        );
    }

    #[test]
    fn accepts_decimal_code_base_but_requires_a_kernel_name() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-elf",
            "kernel.o",
            "--kernel",
            "ClearL2Cache",
            "--aligned-code-base",
            "268439552",
        ])
        .unwrap();
        let Command::InspectElf {
            aligned_code_base, ..
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(aligned_code_base, Some(0x1000_1000));
        assert!(
            Cli::try_parse_from([
                "open-ascend-emulator",
                "inspect-elf",
                "kernel.o",
                "--aligned-code-base",
                "0x1000",
            ])
            .is_err()
        );
    }

    #[test]
    fn parses_bounded_simulator_pc_start_and_rejects_ambiguous_sources() {
        assert_eq!(parse_pc_start_addr(b"0x10d11000"), Ok(0x10d1_1000));
        assert_eq!(parse_pc_start_addr(b"0x10d0d000\n"), Ok(0x10d0_d000));
        assert_eq!(parse_pc_start_addr(b"0x1000\r\n"), Ok(0x1000));
        for invalid in [
            b"".as_slice(),
            b"0".as_slice(),
            b"0x0".as_slice(),
            b" 0x1000".as_slice(),
            b"0x1000\nextra".as_slice(),
            b"0x10000000000000000".as_slice(),
        ] {
            assert!(parse_pc_start_addr(invalid).is_err());
        }

        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-elf",
            "kernel.o",
            "--kernel",
            "ClearL2Cache",
            "--pc-start-addr-file",
            "pc_start_addr.txt",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::InspectElf {
                pc_start_addr_file: Some(ref path),
                aligned_code_base: None,
                ..
            } if path == &PathBuf::from("pc_start_addr.txt")
        ));
        assert!(
            Cli::try_parse_from([
                "open-ascend-emulator",
                "inspect-elf",
                "kernel.o",
                "--pc-start-addr-file",
                "pc_start_addr.txt",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "open-ascend-emulator",
                "inspect-elf",
                "kernel.o",
                "--kernel",
                "ClearL2Cache",
                "--pc-start-addr-file",
                "pc_start_addr.txt",
                "--aligned-code-base",
                "0x1000",
            ])
            .is_err()
        );
    }

    #[test]
    fn raw_word_json_includes_only_present_decoder_hints() {
        let word = RawWord {
            pc: 8,
            device_pc: Some(0x1008),
            word: 0x08c0_0000,
            decoder_hint: AicDecoderHint::from_word(Architecture::Dav3510, 0x08c0_0000),
        };
        let value = serde_json::to_value(word).unwrap();
        assert_eq!(value["decoder_hint"]["ScalarKey8"]["vendor_isa_name"], 49);
        assert_eq!(value["device_pc"], 0x1008);

        let unclassified = RawWord {
            pc: 12,
            device_pc: None,
            word: 0xffff_ffff,
            decoder_hint: None,
        };
        assert!(
            serde_json::to_value(unclassified)
                .unwrap()
                .get("decoder_hint")
                .is_none()
        );
    }

    #[test]
    fn parses_scalar_trace_verification_options() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "verify-scalar-trace",
            "-",
            "--architecture",
            "dav_2201",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::VerifyScalarTrace {
                path,
                architecture: Architecture::Dav2201,
                json: true,
            } if path.as_os_str() == "-"
        ));
    }

    #[test]
    fn parses_acl_argument_inspection_options() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-config",
            "kernel_config.bin",
            "--acl-args",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::InspectConfig {
                path,
                acl_args: true,
                stage_inputs: false,
                max_loaded_bytes: None,
                json: true,
            } if path.as_os_str() == "kernel_config.bin"
        ));

        assert!(
            Cli::try_parse_from([
                "open-ascend-emulator",
                "inspect-config",
                "kernel_config.bin",
                "--stage-inputs"
            ])
            .is_err()
        );
        let staged = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-config",
            "kernel_config.bin",
            "--stage-inputs",
            "--max-loaded-bytes",
            "4096",
        ])
        .unwrap();
        assert!(matches!(
            staged.command,
            Command::InspectConfig {
                stage_inputs: true,
                max_loaded_bytes: Some(4096),
                ..
            }
        ));
    }
}
