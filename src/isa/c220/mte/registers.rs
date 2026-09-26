use super::{
    C220MovInstruction, bias::C220MovL1ToBtInstruction, factor::C220FactorLoadInstruction,
    fixp::C220FixpInstruction, load2d::C220Load2dInstruction,
    load2d_sparse::C220Load2dSparseInstruction, load2d_transpose::C220Load2dTransposeInstruction,
    out_to_l1::C220MovOutToL1Instruction, set2d::C220Set2dInstruction,
};

/// Scalar register operands read by the supported MTE instruction forms.
/// Destination address registers are inputs, not scalar register writes.
/// Returns `None` for an instruction outside these decoded forms.
pub const fn read_register_mask(word: u32) -> Option<u32> {
    let xd = 1 << ((word >> 17) & 31);
    let xn = 1 << ((word >> 12) & 31);
    let xm = 1 << ((word >> 7) & 31);
    let xt = 1 << ((word >> 2) & 31);
    if super::spr::C220Mte1SprWrite::decode(word).is_some() {
        Some(xn)
    } else if C220Set2dInstruction::decode(word).is_some() {
        Some(xd | xm)
    } else if super::smask::C220MovSmaskInstruction::decode(word).is_some() {
        Some(xd | xn | xt)
    } else if C220Load2dTransposeInstruction::decode(word).is_some()
        || C220FixpInstruction::decode(word).is_some()
        || super::load3d::C220Load3dV2Instruction::decode(word).is_some()
    {
        Some(xd | xn | xm | xt)
    } else if C220MovInstruction::decode(word).is_some()
        || C220MovOutToL1Instruction::decode(word).is_some()
        || C220Load2dInstruction::decode(word).is_some()
        || C220Load2dSparseInstruction::decode(word).is_some()
        || C220MovL1ToBtInstruction::decode(word).is_some()
        || C220FactorLoadInstruction::decode(word).is_some()
    {
        Some(xd | xn | xm)
    } else {
        None
    }
}
