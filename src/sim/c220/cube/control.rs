use crate::sim::c220::numeric::fp16::C220Fp16Mode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220F32MmadMode {
    Fp32,
    Hf32,
}

impl C220F32MmadMode {
    const fn from_spr3(value: u64) -> Self {
        if value & (1 << 46) == 0 {
            Self::Fp32
        } else {
            Self::Hf32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeExecutionControl {
    pub fp16_mode: C220Fp16Mode,
    pub f32_mode: C220F32MmadMode,
    pub hf32_rounding: bool,
}

impl C220CubeExecutionControl {
    pub const fn from_spr3(value: u64) -> Self {
        Self {
            fp16_mode: C220Fp16Mode::from_control_spr(value),
            f32_mode: C220F32MmadMode::from_spr3(value),
            hf32_rounding: value & (1 << 47) != 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220CubeIssueDelay {
    pub enabled: bool,
    pub previous_issue_guard_ticks: u8,
    pub first_observation_delay_ticks: u8,
}

impl C220CubeIssueDelay {
    pub const fn from_sprs(spr107: u64, spr108: u64) -> Self {
        Self {
            enabled: spr107 & (1 << 4) != 0,
            previous_issue_guard_ticks: ((spr108 >> 2) & 0x3f) as u8,
            first_observation_delay_ticks: (((spr107 >> 1) & 0x7)
                + ((spr107 >> 5) & 0xf)
                + (((spr108 >> 24) & 0x3) << 4)) as u8,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220CubeTimingControl {
    pub f32_mode: C220F32MmadMode,
    pub fsm_m_priority: bool,
    pub issue_delay: C220CubeIssueDelay,
}

impl C220CubeTimingControl {
    pub const fn from_sprs(spr3: u64, spr107: u64, spr108: u64) -> Self {
        Self {
            f32_mode: C220F32MmadMode::from_spr3(spr3),
            fsm_m_priority: spr3 & (1 << 51) != 0,
            issue_delay: C220CubeIssueDelay::from_sprs(spr107, spr108),
        }
    }
}
