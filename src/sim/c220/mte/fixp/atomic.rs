pub(super) use crate::sim::c220::numeric::atomic::combine_atomic;
use crate::sim::c220::numeric::fp16::C220Fp16AddRounding;

/// Model-level atomic controls, independent of the instruction's captured CTRL.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct C220FixpAtomicConfig {
    pub enabled: bool,
    pub fp16_rounding: C220Fp16AddRounding,
}
