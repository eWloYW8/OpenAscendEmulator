use crate::fp32_vector::{
    Fp32LaneOutcome, Fp32MaskLayout, Fp32VectorError, Fp32VectorOperation,
    evaluate_masked_fp32_lanes,
};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum C220VecArithmeticOperation {
    Add,
    Subtract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct C220VecArithmeticHint {
    pub operation: C220VecArithmeticOperation,
    pub vendor_isa_name: u16,
    pub x_register_index_0: u8,
    pub x_register_index_4: u8,
    pub x_register_index_6: u8,
    pub x_register_index_8: u8,
    pub dtype_selector: u8,
    pub vendor_dtype_code: u8,
}

impl C220VecArithmeticHint {
    pub const fn from_word(word: u32) -> Option<Self> {
        if (word >> 29) != 4 || ((word >> 25) & 0xf) != 2 {
            return None;
        }
        let (operation, vendor_isa_name) = match word & 3 {
            0 => (C220VecArithmeticOperation::Add, 189),
            1 => (C220VecArithmeticOperation::Subtract, 190),
            _ => return None,
        };
        let dtype_selector = ((word >> 22) & 3) as u8;
        let vendor_dtype_code = match dtype_selector {
            0 => 10,
            1 => 8,
            2 => 6,
            _ => 14,
        };
        Some(Self {
            operation,
            vendor_isa_name,
            x_register_index_0: ((word >> 17) & 0x1f) as u8,
            x_register_index_4: ((word >> 12) & 0x1f) as u8,
            x_register_index_6: ((word >> 7) & 0x1f) as u8,
            x_register_index_8: ((word >> 2) & 0x1f) as u8,
            dtype_selector,
            vendor_dtype_code,
        })
    }

    pub const fn has_fp32_value_path(self) -> bool {
        self.vendor_dtype_code == 14
    }

    pub fn evaluate_fp32_lanes(
        self,
        first: &[u32],
        second: &[u32],
        iteration_mask: &[u64; 4],
    ) -> Result<Vec<Fp32LaneOutcome>, Fp32VectorError> {
        if !self.has_fp32_value_path() {
            return Err(Fp32VectorError::UnsupportedInstruction);
        }
        let operation = match self.operation {
            C220VecArithmeticOperation::Add => Fp32VectorOperation::Add,
            C220VecArithmeticOperation::Subtract => Fp32VectorOperation::Subtract,
        };
        evaluate_masked_fp32_lanes(
            operation,
            Fp32MaskLayout::C220Lane,
            first,
            second,
            iteration_mask,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_add_and_sub_words_select_distinct_vec_handlers() {
        let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
        let subtract = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
        assert_eq!(add.operation, C220VecArithmeticOperation::Add);
        assert_eq!(subtract.operation, C220VecArithmeticOperation::Subtract);
        assert_eq!(add.vendor_isa_name, 189);
        assert_eq!(subtract.vendor_isa_name, 190);
        assert_eq!(add.x_register_index_0, 14);
        assert_eq!(add.x_register_index_4, 11);
        assert_eq!(add.x_register_index_6, 12);
        assert_eq!(add.x_register_index_8, 6);
        assert_eq!(add.dtype_selector, 3);
        assert_eq!(add.vendor_dtype_code, 14);
        assert!(add.has_fp32_value_path());
        assert!(subtract.has_fp32_value_path());
    }

    #[test]
    fn rejects_other_route_or_leaf() {
        let add = 0x85dc_b618;
        for wrong in [add ^ (1 << 29), add ^ (1 << 25), add | 2, add | 3] {
            assert_eq!(C220VecArithmeticHint::from_word(wrong), None);
        }
    }

    #[test]
    fn dtype_table_matches_vendor_constants() {
        let base = 0x85dc_b618 & !(3 << 22);
        for (selector, code) in [(0, 10), (1, 8), (2, 6), (3, 14)] {
            let hint = C220VecArithmeticHint::from_word(base | (selector << 22)).unwrap();
            assert_eq!(hint.dtype_selector, selector as u8);
            assert_eq!(hint.vendor_dtype_code, code);
        }
    }

    #[test]
    fn captured_fp32_words_reach_the_masked_value_stage() {
        let first = [1.0_f32.to_bits(), 2.0_f32.to_bits()];
        let second = [3.0_f32.to_bits(), 4.0_f32.to_bits()];
        let mask = [1, 0, 0, 0];
        let add = C220VecArithmeticHint::from_word(0x85dc_b618).unwrap();
        let sub = C220VecArithmeticHint::from_word(0x85dc_b619).unwrap();
        let added = add.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        let subtracted = sub.evaluate_fp32_lanes(&first, &second, &mask).unwrap();
        assert_eq!(added[0].bits, 4.0_f32.to_bits());
        assert_eq!(subtracted[0].bits, (-2.0_f32).to_bits());
        assert_eq!(added[1].bits, 0);
        assert_eq!(subtracted[1].bits, 0);
    }
}
