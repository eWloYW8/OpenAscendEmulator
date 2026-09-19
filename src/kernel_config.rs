use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const KNOWN_KERNEL_CONFIG_FIELDS: &[&str] = &[
    "bin_path",
    "block_dim",
    "device_id",
    "ffts",
    "input_path",
    "input_size",
    "kernel_name",
    "magic",
    "output_dir",
    "output_name",
    "output_size",
    "workspace_size",
    "tiling_data_path",
    "tiling_key",
    "old_mode",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelConfigDocument {
    pub fields: BTreeMap<String, String>,
}

impl KernelConfigDocument {
    pub fn from_slice(bytes: &[u8]) -> Result<Self, KernelConfigError> {
        let value: Value = serde_json::from_slice(bytes)?;
        let Value::Object(entries) = value else {
            return Err(KernelConfigError::RootNotObject);
        };

        let mut fields = BTreeMap::new();
        for (key, value) in entries {
            let Value::String(value) = value else {
                return Err(KernelConfigError::FieldNotString { field: key });
            };
            fields.insert(key, value);
        }
        Ok(Self { fields })
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, KernelConfigError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(|source| KernelConfigError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_slice(&bytes)
    }

    pub fn to_json_vec(&self) -> Result<Vec<u8>, KernelConfigError> {
        Ok(serde_json::to_vec(&self.fields)?)
    }

    pub fn decode(&self) -> Result<DecodedKernelConfig, KernelConfigError> {
        let fields = &self.fields;
        let block_dim = parse_optional_i32(fields, "block_dim")?.unwrap_or_default();
        let device_id = parse_optional_i32(fields, "device_id")?.unwrap_or_default();

        let kernel_name = fields.get("kernel_name").cloned().unwrap_or_default();
        if fields.contains_key("kernel_name") && !valid_kernel_name(&kernel_name) {
            return Err(KernelConfigError::InvalidKernelName(kernel_name));
        }

        let magic = fields
            .get("magic")
            .map(|value| BinaryMagic::parse(value))
            .transpose()?;
        let inputs = decode_files(fields, "input_path", "input_size", true)?;
        let outputs = decode_files(fields, "output_name", "output_size", false)?;
        let workspace_sizes = fields
            .get("workspace_size")
            .map(|value| parse_u64_list("workspace_size", value))
            .transpose()?
            .unwrap_or_default();
        let tiling_data_parts = fields
            .get("tiling_data_path")
            .map(|value| split_vendor_list(value))
            .unwrap_or_default();
        let tiling_data = if fields
            .get("tiling_data_path")
            .is_some_and(|value| !value.is_empty())
        {
            if tiling_data_parts.len() < 2 {
                return Err(KernelConfigError::MalformedTilingDataPath {
                    parts: tiling_data_parts.len(),
                });
            }
            Some(ReplayTilingData {
                path: tiling_data_parts[0].clone(),
                size: parse_vendor_u64("tiling_data_path", &tiling_data_parts[1])?,
                ignored_parts: tiling_data_parts[2..].to_vec(),
            })
        } else {
            None
        };
        let tiling_key = parse_optional_u64(fields, "tiling_key")?;

        let runner = match fields.get("old_mode") {
            None => ReplayRunner::Legacy,
            Some(value) if value == "1" => ReplayRunner::Legacy,
            Some(_) => ReplayRunner::Acl,
        };
        let ffts = fields
            .get("ffts")
            .is_some_and(|value| value.as_bytes().first() == Some(&b'Y'));

        let present_fields = fields.keys().cloned().collect();
        let unknown_fields = fields
            .iter()
            .filter(|(key, _)| !KNOWN_KERNEL_CONFIG_FIELDS.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();

        Ok(DecodedKernelConfig {
            bin_path: fields.get("bin_path").cloned().unwrap_or_default(),
            block_dim,
            device_id,
            ffts,
            inputs,
            kernel_name,
            magic,
            output_dir: fields.get("output_dir").cloned().unwrap_or_default(),
            outputs,
            workspace_sizes,
            tiling_data,
            tiling_data_parts,
            tiling_key,
            runner,
            present_fields,
            unknown_fields,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayRunner {
    Legacy,
    Acl,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BinaryMagic {
    #[serde(rename = "RT_DEV_BINARY_MAGIC_ELF")]
    Elf,
    #[serde(rename = "RT_DEV_BINARY_MAGIC_ELF_AICUBE")]
    AiCube,
    #[serde(rename = "RT_DEV_BINARY_MAGIC_ELF_AIVEC")]
    AiVec,
}

impl BinaryMagic {
    pub fn parse(value: &str) -> Result<Self, KernelConfigError> {
        match value {
            "RT_DEV_BINARY_MAGIC_ELF" => Ok(Self::Elf),
            "RT_DEV_BINARY_MAGIC_ELF_AICUBE" => Ok(Self::AiCube),
            "RT_DEV_BINARY_MAGIC_ELF_AIVEC" => Ok(Self::AiVec),
            _ => Err(KernelConfigError::InvalidMagic(value.to_owned())),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Elf => "RT_DEV_BINARY_MAGIC_ELF",
            Self::AiCube => "RT_DEV_BINARY_MAGIC_ELF_AICUBE",
            Self::AiVec => "RT_DEV_BINARY_MAGIC_ELF_AIVEC",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayFile {
    pub path: String,
    pub size: Option<u64>,
    pub is_null_input: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayTilingData {
    pub path: String,
    pub size: u64,
    pub ignored_parts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DecodedKernelConfig {
    pub bin_path: String,
    pub block_dim: i32,
    pub device_id: i32,
    pub ffts: bool,
    pub inputs: Vec<ReplayFile>,
    pub kernel_name: String,
    pub magic: Option<BinaryMagic>,
    pub output_dir: String,
    pub outputs: Vec<ReplayFile>,
    pub workspace_sizes: Vec<u64>,
    pub tiling_data: Option<ReplayTilingData>,
    pub tiling_data_parts: Vec<String>,
    pub tiling_key: Option<u64>,
    pub runner: ReplayRunner,
    pub present_fields: Vec<String>,
    pub unknown_fields: BTreeMap<String, String>,
}

#[derive(Debug, Error)]
pub enum KernelConfigError {
    #[error("failed to access kernel config {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid kernel config JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("kernel config root must be a JSON object")]
    RootNotObject,
    #[error("kernel config field {field} must be a string")]
    FieldNotString { field: String },
    #[error("field {field} is not a valid {kind}: {value}")]
    InvalidInteger {
        field: &'static str,
        kind: &'static str,
        value: String,
    },
    #[error("kernel_name must contain 1..=255 ASCII alphanumeric or underscore bytes: {0}")]
    InvalidKernelName(String),
    #[error("invalid runtime binary magic: {0}")]
    InvalidMagic(String),
    #[error("{field} does not contain any elements")]
    EmptyList { field: &'static str },
    #[error("{size_field} has {actual} elements but {path_field} has {expected}")]
    CountMismatch {
        path_field: &'static str,
        size_field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("tiling_data_path requires at least path and size, got {parts} field(s)")]
    MalformedTilingDataPath { parts: usize },
}

fn decode_files(
    fields: &BTreeMap<String, String>,
    path_field: &'static str,
    size_field: &'static str,
    reject_explicit_empty_paths: bool,
) -> Result<Vec<ReplayFile>, KernelConfigError> {
    let paths = fields
        .get(path_field)
        .map(|value| split_vendor_list(value))
        .unwrap_or_default();
    if reject_explicit_empty_paths && fields.contains_key(path_field) && paths.is_empty() {
        return Err(KernelConfigError::EmptyList { field: path_field });
    }

    let sizes = fields
        .get(size_field)
        .map(|value| parse_u64_list(size_field, value))
        .transpose()?;
    if let Some(sizes) = &sizes
        && sizes.len() != paths.len()
    {
        return Err(KernelConfigError::CountMismatch {
            path_field,
            size_field,
            expected: paths.len(),
            actual: sizes.len(),
        });
    }

    Ok(paths
        .into_iter()
        .enumerate()
        .map(|(index, path)| ReplayFile {
            is_null_input: reject_explicit_empty_paths && path == "n",
            path,
            size: sizes.as_ref().map(|sizes| sizes[index]),
        })
        .collect())
}

fn parse_optional_i32(
    fields: &BTreeMap<String, String>,
    field: &'static str,
) -> Result<Option<i32>, KernelConfigError> {
    fields
        .get(field)
        .map(|value| parse_vendor_i32(field, value))
        .transpose()
}

fn parse_optional_u64(
    fields: &BTreeMap<String, String>,
    field: &'static str,
) -> Result<Option<u64>, KernelConfigError> {
    fields
        .get(field)
        .filter(|value| !value.is_empty())
        .map(|value| parse_vendor_u64(field, value))
        .transpose()
}

fn parse_u64_list(field: &'static str, value: &str) -> Result<Vec<u64>, KernelConfigError> {
    split_vendor_list(value)
        .into_iter()
        .map(|part| parse_vendor_u64(field, &part))
        .collect()
}

fn parse_vendor_i32(field: &'static str, value: &str) -> Result<i32, KernelConfigError> {
    if !vendor_integer_syntax(value) {
        return Err(invalid_integer(field, "signed 32-bit integer", value));
    }
    let value_without_plus = value.strip_prefix('+').unwrap_or(value);
    value_without_plus
        .parse()
        .map_err(|_| invalid_integer(field, "signed 32-bit integer", value))
}

fn parse_vendor_u64(field: &'static str, value: &str) -> Result<u64, KernelConfigError> {
    if !vendor_integer_syntax(value) {
        return Err(invalid_integer(field, "unsigned 64-bit integer", value));
    }
    let (negative, magnitude) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else {
        (false, value.strip_prefix('+').unwrap_or(value))
    };
    let magnitude: u64 = magnitude
        .parse()
        .map_err(|_| invalid_integer(field, "unsigned 64-bit integer", value))?;
    Ok(if negative {
        0_u64.wrapping_sub(magnitude)
    } else {
        magnitude
    })
}

fn invalid_integer(field: &'static str, kind: &'static str, value: &str) -> KernelConfigError {
    KernelConfigError::InvalidInteger {
        field,
        kind,
        value: value.to_owned(),
    }
}

fn vendor_integer_syntax(value: &str) -> bool {
    let bytes = value.as_bytes();
    let digits = match bytes.first() {
        Some(b'+') | Some(b'-') => &bytes[1..],
        Some(_) => bytes,
        None => return false,
    };
    !digits.is_empty() && digits.iter().all(u8::is_ascii_digit)
}

fn valid_kernel_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 255
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn split_vendor_list(value: &str) -> Vec<String> {
    if value.is_empty() {
        Vec::new()
    } else {
        value.split_terminator(';').map(str::to_owned).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_injection_style_acl_replay_config() {
        let document = KernelConfigDocument::from_slice(
            br#"{
                "bin_path":"/run/kernel_data/aicore_binary.o",
                "block_dim":"8",
                "device_id":"0",
                "ffts":"Y",
                "input_path":"input0.bin;input1.bin",
                "input_size":"64;128",
                "kernel_name":"add_custom_1",
                "magic":"RT_DEV_BINARY_MAGIC_ELF_AIVEC",
                "tiling_key":"17",
                "old_mode":"0"
            }"#,
        )
        .unwrap();
        let config = document.decode().unwrap();

        assert_eq!(config.runner, ReplayRunner::Acl);
        assert_eq!(config.block_dim, 8);
        assert!(config.ffts);
        assert_eq!(config.inputs[0].size, Some(64));
        assert_eq!(config.inputs[1].path, "input1.bin");
        assert!(!config.inputs[0].is_null_input);
        assert!(config.tiling_data.is_none());
        assert_eq!(config.magic, Some(BinaryMagic::AiVec));
        assert_eq!(config.tiling_key, Some(17));
        assert!(config.unknown_fields.is_empty());
    }

    #[test]
    fn old_mode_defaults_to_legacy_and_only_exact_one_selects_it() {
        let absent = KernelConfigDocument::from_slice(br#"{}"#)
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(absent.runner, ReplayRunner::Legacy);

        let one = KernelConfigDocument::from_slice(br#"{"old_mode":"1"}"#)
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(one.runner, ReplayRunner::Legacy);

        let other = KernelConfigDocument::from_slice(br#"{"old_mode":"true"}"#)
            .unwrap()
            .decode()
            .unwrap();
        assert_eq!(other.runner, ReplayRunner::Acl);
    }

    #[test]
    fn ffts_matches_launcher_first_byte_check() {
        for (value, expected) in [("Y", true), ("Yes", true), ("N", false), ("1", false)] {
            let mut fields = BTreeMap::new();
            fields.insert("ffts".into(), value.into());
            assert_eq!(
                KernelConfigDocument { fields }.decode().unwrap().ffts,
                expected
            );
        }
    }

    #[test]
    fn every_json_value_must_be_a_string() {
        assert!(matches!(
            KernelConfigDocument::from_slice(br#"{"block_dim":8}"#),
            Err(KernelConfigError::FieldNotString { field }) if field == "block_dim"
        ));
    }

    #[test]
    fn input_paths_and_sizes_must_have_equal_counts() {
        let document =
            KernelConfigDocument::from_slice(br#"{"input_path":"a;b","input_size":"4"}"#).unwrap();
        assert!(matches!(
            document.decode(),
            Err(KernelConfigError::CountMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
    }

    #[test]
    fn unknown_fields_are_retained_for_evidence() {
        let document = KernelConfigDocument::from_slice(br#"{"future_key":"value"}"#).unwrap();
        let decoded = document.decode().unwrap();
        assert_eq!(decoded.unknown_fields["future_key"], "value");
        assert_eq!(
            document.to_json_vec().unwrap(),
            br#"{"future_key":"value"}"#
        );
    }

    #[test]
    fn integer_parser_matches_vendor_sign_syntax() {
        assert_eq!(parse_vendor_i32("device_id", "+7").unwrap(), 7);
        assert_eq!(parse_vendor_i32("device_id", "-7").unwrap(), -7);
        assert_eq!(parse_vendor_u64("input_size", "-1").unwrap(), u64::MAX);
        assert!(parse_vendor_u64("input_size", " 1").is_err());
        assert!(parse_vendor_u64("input_size", "1x").is_err());
    }

    #[test]
    fn decodes_null_input_and_tiling_path_size_without_losing_raw_parts() {
        let document = KernelConfigDocument::from_slice(
            br#"{"old_mode":"0","input_path":"n;input.bin","input_size":"999;4","tiling_data_path":"tiling.bin;96;ignored"}"#,
        )
        .unwrap();
        let config = document.decode().unwrap();
        assert!(config.inputs[0].is_null_input);
        assert_eq!(config.inputs[0].size, Some(999));
        assert!(!config.inputs[1].is_null_input);
        assert_eq!(config.tiling_data.as_ref().unwrap().path, "tiling.bin");
        assert_eq!(config.tiling_data.as_ref().unwrap().size, 96);
        assert_eq!(
            config.tiling_data.as_ref().unwrap().ignored_parts,
            ["ignored"]
        );
        assert_eq!(config.tiling_data_parts, ["tiling.bin", "96", "ignored"]);
    }

    #[test]
    fn rejects_tiling_path_without_size_or_with_invalid_size() {
        let path_only =
            KernelConfigDocument::from_slice(br#"{"tiling_data_path":"tile.bin"}"#).unwrap();
        assert!(matches!(
            path_only.decode(),
            Err(KernelConfigError::MalformedTilingDataPath { parts: 1 })
        ));
        let bad_size =
            KernelConfigDocument::from_slice(br#"{"tiling_data_path":"tile.bin;wrong"}"#).unwrap();
        assert!(matches!(
            bad_size.decode(),
            Err(KernelConfigError::InvalidInteger {
                field: "tiling_data_path",
                ..
            })
        ));
    }

    #[test]
    fn null_input_marker_is_case_sensitive_and_output_name_is_not_an_input() {
        let config = KernelConfigDocument::from_slice(
            br#"{"input_path":"n;N","input_size":"1;1","output_name":"n","output_size":"1"}"#,
        )
        .unwrap()
        .decode()
        .unwrap();
        assert!(config.inputs[0].is_null_input);
        assert!(!config.inputs[1].is_null_input);
        assert!(!config.outputs[0].is_null_input);
    }
}
