use crate::acl_args::{AclArgumentPlan, AclArgumentPlanError};
use crate::architecture::{Architecture, all_device_profiles};
use crate::device_elf::{
    DeviceElf, DeviceElfError, DeviceElfHeader, DeviceGlobalPatchSite, DeviceKernelSummary,
    DeviceLoadImageSummary,
};
use crate::flow_trace::verify_jump_trace;
use crate::isa::{AicDecoderHint, AicFramingError, AicInstructionWord, AicWordFramer};
use crate::kernel_config::{KernelConfigDocument, KernelConfigError};
use crate::plan::{LaunchPlan, PlanError, SimulatorRequest};
use crate::prof_stub_flow::inspect_prof_stub_branch_edges;
use crate::prof_stub_object_verify::{ProfStubObjectVerification, verify_prof_stub_object};
use crate::prof_stub_packet::ProfStubPacketError;
use crate::prof_stub_stream::inspect_prof_stub_stream;
use crate::replay_seed::{ReplaySeed, ReplaySeedError};
use crate::rvec::C310RvecArithmeticHint;
use crate::trace::verify_scalar_trace;
use crate::vec_c220::C220VecArithmeticHint;
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

fn parse_u32_word(raw: &str) -> Result<u32, String> {
    let value = parse_u64_address(raw)?;
    u32::try_from(value).map_err(|_| format!("instruction word exceeds 32 bits: {raw}"))
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

fn read_bounded_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>, CliError> {
    let mut bytes = Vec::new();
    let file = File::open(path).map_err(|source| CliError::FileRead {
        path: path.to_path_buf(),
        source,
    })?;
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|source| CliError::FileRead {
            path: path.to_path_buf(),
            source,
        })?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(CliError::InputFileTooLarge {
            path: path.to_path_buf(),
            limit: max_bytes,
        });
    }
    Ok(bytes)
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
    InspectRvecWord {
        #[arg(value_parser = parse_u32_word)]
        word: u32,
        #[arg(long)]
        json: bool,
    },
    InspectC220VecWord {
        #[arg(value_parser = parse_u32_word)]
        word: u32,
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
    VerifyJumpTrace {
        path: PathBuf,
        #[arg(long)]
        architecture: Architecture,
        #[arg(long)]
        json: bool,
    },
    InspectProfStubStream {
        path: PathBuf,
        #[arg(long, default_value_t = 1_048_576)]
        max_bytes: u64,
        #[arg(long, default_value_t = 16)]
        max_records: usize,
        #[arg(long)]
        json: bool,
    },
    InspectProfStubBranchEdges {
        path: PathBuf,
        #[arg(long)]
        architecture: Architecture,
        #[arg(long, default_value_t = 1_048_576)]
        max_bytes: u64,
        #[arg(long, default_value_t = 16)]
        max_issues: usize,
        #[arg(long)]
        json: bool,
    },
    VerifyProfStubObject {
        stream: PathBuf,
        #[arg(long)]
        object: PathBuf,
        #[arg(long)]
        kernel: String,
        #[arg(long)]
        pc_start_addr_file: PathBuf,
        #[arg(long, default_value_t = 1_048_576)]
        max_bytes: u64,
        #[arg(long, default_value_t = 16_777_216)]
        max_object_bytes: u64,
        #[arg(long, default_value_t = 16)]
        max_examples: usize,
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
    #[error(transparent)]
    ProfStubPacket(#[from] ProfStubPacketError),
    #[error("input file {path} exceeds explicit {limit}-byte read limit")]
    InputFileTooLarge { path: PathBuf, limit: u64 },
    #[error(
        "ProfStub object binding did not verify all instruction records: {checked}/{total} checked"
    )]
    ProfStubObjectUnverified { checked: usize, total: usize },
    #[error("failed to read scalar trace: {0}")]
    TraceRead(#[from] io::Error),
    #[error("scalar trace contained no supported scalar arithmetic, move, or ZEROEXT records")]
    EmptyScalarTrace,
    #[error("scalar trace has {0} mismatches")]
    ScalarTraceMismatch(u64),
    #[error("jump trace contained no JUMP records")]
    EmptyJumpTrace,
    #[error("jump trace has {0} mismatches")]
    JumpTraceMismatch(u64),
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
        Command::InspectRvecWord { word, json } => {
            let output = RvecWordInspection {
                word,
                architecture: Architecture::Dav3510,
                hint: C310RvecArithmeticHint::from_word(word),
            };
            if json {
                print_json(&output)?;
            } else if let Some(hint) = output.hint {
                println!("word: {:#010x}", output.word);
                println!("operation: {:?}", hint.operation);
                println!("vendor-isa-name: {}", hint.vendor_isa_name);
                println!("destination-v-register: {}", hint.destination_v_register);
                println!("first-source-v-register: {}", hint.first_source_v_register);
                println!(
                    "second-source-v-register: {}",
                    hint.second_source_v_register
                );
                println!("predicate-register: {}", hint.predicate_register);
                println!("dtype-selector: {}", hint.dtype_selector);
                println!("fp32-value-path: {}", hint.has_fp32_value_path());
            } else {
                println!("word: {:#010x}", output.word);
                println!("C310 RVec VADD/VSUB opcode: unrecognized");
            }
        }
        Command::InspectC220VecWord { word, json } => {
            let output = C220VecWordInspection {
                word,
                architecture: Architecture::Dav2201,
                hint: C220VecArithmeticHint::from_word(word),
            };
            if json {
                print_json(&output)?;
            } else if let Some(hint) = output.hint {
                println!("word: {:#010x}", output.word);
                println!("operation: {:?}", hint.operation);
                println!("vendor-isa-name: {}", hint.vendor_isa_name);
                println!("x-register-index-0: {}", hint.x_register_index_0);
                println!("x-register-index-4: {}", hint.x_register_index_4);
                println!("x-register-index-6: {}", hint.x_register_index_6);
                println!("x-register-index-8: {}", hint.x_register_index_8);
                println!("dtype-selector: {}", hint.dtype_selector);
                println!("vendor-dtype-code: {}", hint.vendor_dtype_code);
                println!("fp32-value-path: {}", hint.has_fp32_value_path());
            } else {
                println!("word: {:#010x}", output.word);
                println!("C220 Vec VADD/VSUB opcode: unrecognized");
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
            let (address_meta_flags, address_meta_error) = match elf.address_meta_flags() {
                Ok(flags) => (Some(flags), None),
                Err(error) => (None, Some(error.to_string())),
            };
            let (global_patch_sites, global_patch_sites_error) = match elf.global_patch_sites() {
                Ok(sites) => (Some(sites), None),
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
                address_meta_flags,
                address_meta_error,
                global_patch_sites,
                global_patch_sites_error,
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
                if let Some(flags) = inspection.address_meta_flags {
                    println!("address-meta-flags: {flags:#x}");
                } else if let Some(error) = &inspection.address_meta_error {
                    println!("address-meta-flags: unavailable ({error})");
                }
                if let Some(sites) = &inspection.global_patch_sites {
                    for site in sites {
                        println!(
                            "global-patch-site: {} file-offset={:#x} image-offset={:#x}",
                            site.symbol.elf_name(),
                            site.file_offset,
                            site.image_offset
                        );
                    }
                } else if let Some(error) = &inspection.global_patch_sites_error {
                    println!("global-patch-sites: unavailable ({error})");
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
                println!("subtract-register: {}", summary.subtract_register);
                println!("multiply-register: {}", summary.multiply_register);
                println!("multiply-add-fields: {}", summary.multiply_add_fields);
                println!(
                    "multiply-add-value-checked: {}",
                    summary.multiply_add_value_checked
                );
                println!(
                    "multiply-add-unknown-prior: {}",
                    summary.multiply_add_unknown_prior
                );
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
                println!("shift-right-fields: {}", summary.shift_right_fields);
                println!(
                    "shift-right-value-checked: {}",
                    summary.shift_right_value_checked
                );
                println!(
                    "shift-right-unknown-prior: {}",
                    summary.shift_right_unknown_prior
                );
                println!("move-immediate: {}", summary.checked_move_immediate);
                println!("move-keep-lane: {}", summary.checked_move_keep_lane);
                println!("move-register: {}", summary.checked_register_move);
                println!("negate: {}", summary.checked_negate);
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
                + summary.subtract_register
                + summary.multiply_register
                + summary.multiply_add_fields
                + summary.and_register
                + summary.or_register
                + summary.shift_left_fields
                + summary.shift_right_fields
                + summary.checked_move_immediate
                + summary.checked_move_keep_lane
                + summary.checked_register_move
                + summary.checked_negate
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
        Command::VerifyJumpTrace {
            path,
            architecture,
            json,
        } => {
            let summary = if path.as_os_str() == "-" {
                verify_jump_trace(io::stdin().lock(), architecture)?
            } else {
                let file = File::open(&path).map_err(|source| CliError::FileRead {
                    path: path.clone(),
                    source,
                })?;
                verify_jump_trace(BufReader::new(file), architecture)?
            };
            if json {
                print_json(&summary)?;
            } else {
                println!("architecture: {}", summary.architecture);
                println!("lines: {}", summary.total_lines);
                println!("jumps: {}", summary.observed_jumps);
                println!("conditional-jumps: {}", summary.observed_conditional_jumps);
                println!("compare-jumps: {}", summary.observed_compare_jumps);
                println!("conditional-taken: {}", summary.conditional_taken);
                println!("conditional-not-taken: {}", summary.conditional_not_taken);
                println!("compare-taken: {}", summary.compare_taken);
                println!("compare-not-taken: {}", summary.compare_not_taken);
                println!("compare-signed: {}", summary.compare_signed);
                println!("compare-unsigned: {}", summary.compare_unsigned);
                println!(
                    "compare-immediate-operands: {}",
                    summary.compare_immediate_operands
                );
                println!(
                    "compare-register-operands: {}",
                    summary.compare_register_operands
                );
                println!("verified-immediate: {}", summary.verified_immediate_targets);
                println!("verified-register: {}", summary.verified_register_targets);
                println!(
                    "unverified-register: {}",
                    summary.unverified_register_targets
                );
                println!("mismatches: {}", summary.mismatches);
                for issue in &summary.issues {
                    println!("  line {}: {}", issue.line_number, issue.reason);
                }
            }
            if summary.observed_jumps == 0 {
                return Err(CliError::EmptyJumpTrace);
            }
            if summary.mismatches != 0 {
                return Err(CliError::JumpTraceMismatch(summary.mismatches));
            }
        }
        Command::InspectProfStubStream {
            path,
            max_bytes,
            max_records,
            json,
        } => {
            let bytes = read_bounded_file(&path, max_bytes)?;
            let summary = inspect_prof_stub_stream(&bytes, max_records)?;
            if json {
                print_json(&summary)?;
            } else {
                println!("stream-bytes: {}", summary.stream_bytes);
                println!("packets: {}", summary.packet_count);
                for (packet_type, count) in &summary.counts_by_type {
                    println!("type[{packet_type}]: {count}");
                }
                for record in &summary.records {
                    println!(
                        "offset={} type={} payload={} {:?}",
                        record.offset, record.packet_type, record.payload_bytes, record.detail
                    );
                }
                println!("omitted-records: {}", summary.omitted_records);
            }
        }
        Command::InspectProfStubBranchEdges {
            path,
            architecture,
            max_bytes,
            max_issues,
            json,
        } => {
            let bytes = read_bounded_file(&path, max_bytes)?;
            let summary = inspect_prof_stub_branch_edges(&bytes, architecture, max_issues)?;
            if json {
                print_json(&summary)?;
            } else {
                println!("architecture: {}", summary.architecture);
                println!(
                    "type-21-instructions: {}",
                    summary.type21_instruction_events
                );
                println!("branch-records: {}", summary.branch_records);
                println!("unconditional-records: {}", summary.unconditional_records);
                println!("conditional-records: {}", summary.conditional_records);
                println!("compare-records: {}", summary.compare_records);
                println!("unconditional-matches: {}", summary.unconditional_matches);
                println!("taken-candidates: {}", summary.conditional_taken_candidates);
                println!(
                    "fallthrough-candidates: {}",
                    summary.conditional_fallthrough_candidates
                );
                println!(
                    "ambiguous-candidates: {}",
                    summary.conditional_ambiguous_candidates
                );
                println!(
                    "register-offset-unavailable: {}",
                    summary.register_offset_unavailable
                );
                println!(
                    "outside-candidate-edges: {}",
                    summary.outside_candidate_edges
                );
                println!("no-successor: {}", summary.no_successor);
                println!(
                    "malformed-branch-records: {}",
                    summary.malformed_branch_records
                );
                for issue in &summary.issues {
                    println!("  {:?}", issue);
                }
                println!("omitted-issues: {}", summary.omitted_issues);
            }
        }
        Command::VerifyProfStubObject {
            stream,
            object,
            kernel,
            pc_start_addr_file,
            max_bytes,
            max_object_bytes,
            max_examples,
            json,
        } => {
            let stream_bytes = read_bounded_file(&stream, max_bytes)?;
            let object_bytes = read_bounded_file(&object, max_object_bytes)?;
            let elf = DeviceElf::parse(&object_bytes)?;
            let code_base = read_pc_start_addr(&pc_start_addr_file)?;
            let projected = elf.project_kernel(&kernel, code_base)?;
            let result = verify_prof_stub_object(&stream_bytes, &projected, max_examples)?;
            let output = ProfStubObjectInspection {
                stream,
                object,
                kernel,
                pc_start_addr_file,
                device_entry_address: projected.entry_address,
                verification: result,
            };
            if json {
                print_json(&output)?;
            } else {
                println!("kernel: {}", output.kernel);
                println!("device-entry: {:#x}", output.device_entry_address);
                println!(
                    "instruction-events: {}",
                    output.verification.instruction_events
                );
                println!("checked-events: {}", output.verification.checked_events);
                println!(
                    "unique-checked-pcs: {}",
                    output.verification.unique_checked_pcs
                );
                println!(
                    "all-instruction-words-match-object: {}",
                    output.verification.all_instruction_words_match_object
                );
                for issue in &output.verification.examples {
                    println!(
                        "  offset={} pc={:#x} {:?}",
                        issue.packet_offset, issue.pc, issue
                    );
                }
                println!("omitted-examples: {}", output.verification.omitted_examples);
            }
            if !output.verification.all_instruction_words_match_object {
                return Err(CliError::ProfStubObjectUnverified {
                    checked: output.verification.checked_events,
                    total: output.verification.instruction_events,
                });
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
struct RvecWordInspection {
    word: u32,
    architecture: Architecture,
    hint: Option<C310RvecArithmeticHint>,
}

#[derive(Debug, Serialize)]
struct C220VecWordInspection {
    word: u32,
    architecture: Architecture,
    hint: Option<C220VecArithmeticHint>,
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
    address_meta_flags: Option<u32>,
    address_meta_error: Option<String>,
    global_patch_sites: Option<Vec<DeviceGlobalPatchSite>>,
    global_patch_sites_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code_base: Option<CodeBaseEvidence>,
    kernels: Vec<DeviceKernelSummary>,
    selected: Option<KernelPreview>,
}

#[derive(Debug, Serialize)]
struct ProfStubObjectInspection {
    stream: PathBuf,
    object: PathBuf,
    kernel: String,
    pc_start_addr_file: PathBuf,
    device_entry_address: u64,
    verification: ProfStubObjectVerification,
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
    fn parses_bounded_prof_stub_stream_inspection() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-prof-stub-stream",
            "capture.bin",
            "--max-bytes",
            "32768",
            "--max-records",
            "0",
            "--json",
        ])
        .unwrap();
        let Command::InspectProfStubStream {
            path,
            max_bytes,
            max_records,
            json,
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(path, PathBuf::from("capture.bin"));
        assert_eq!(max_bytes, 32768);
        assert_eq!(max_records, 0);
        assert!(json);
    }

    #[test]
    fn parses_bounded_prof_stub_branch_edge_inspection() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-prof-stub-branch-edges",
            "capture.bin",
            "--architecture",
            "dav_3510",
            "--max-bytes",
            "10000000",
            "--max-issues",
            "3",
            "--json",
        ])
        .unwrap();
        let Command::InspectProfStubBranchEdges {
            path,
            architecture,
            max_bytes,
            max_issues,
            json,
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(path, PathBuf::from("capture.bin"));
        assert_eq!(architecture, Architecture::Dav3510);
        assert_eq!(max_bytes, 10_000_000);
        assert_eq!(max_issues, 3);
        assert!(json);
    }

    #[test]
    fn parses_object_bound_prof_stub_verification() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "verify-prof-stub-object",
            "capture.bin",
            "--object",
            "kernel.o",
            "--kernel",
            "ClearL2Cache",
            "--pc-start-addr-file",
            "pc_start_addr.txt",
            "--max-bytes",
            "32768",
            "--max-object-bytes",
            "8388608",
            "--max-examples",
            "4",
            "--json",
        ])
        .unwrap();
        let Command::VerifyProfStubObject {
            stream,
            object,
            kernel,
            pc_start_addr_file,
            max_bytes,
            max_object_bytes,
            max_examples,
            json,
        } = cli.command
        else {
            panic!("unexpected command")
        };
        assert_eq!(stream, PathBuf::from("capture.bin"));
        assert_eq!(object, PathBuf::from("kernel.o"));
        assert_eq!(kernel, "ClearL2Cache");
        assert_eq!(pc_start_addr_file, PathBuf::from("pc_start_addr.txt"));
        assert_eq!(max_bytes, 32768);
        assert_eq!(max_object_bytes, 8_388_608);
        assert_eq!(max_examples, 4);
        assert!(json);
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
    fn parses_bounded_c310_rvec_word() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-rvec-word",
            "0x80082781",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::InspectRvecWord {
                word: 0x8008_2781,
                json: true
            }
        ));
        assert!(
            Cli::try_parse_from(["open-ascend-emulator", "inspect-rvec-word", "0x100000000"])
                .is_err()
        );
    }

    #[test]
    fn parses_bounded_c220_vec_word() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "inspect-c220-vec-word",
            "0x85dcb619",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::InspectC220VecWord {
                word: 0x85dc_b619,
                json: true
            }
        ));
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
    fn parses_jump_trace_verification_options() {
        let cli = Cli::try_parse_from([
            "open-ascend-emulator",
            "verify-jump-trace",
            "-",
            "--architecture",
            "dav_3510",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::VerifyJumpTrace {
                path,
                architecture: Architecture::Dav3510,
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
