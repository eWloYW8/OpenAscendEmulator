use crate::instruction::c310::layout::C310_PB_SLOT_BYTES;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310RvecScalarWrite {
    pub register_index: u8,
    pub value: u32,
    pub source_payload_word: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310PbRvecScalarProjection {
    pub big_flags: u32,
    pub consumed_payload_words: u8,
    pub writes: Vec<C310RvecScalarWrite>,
}

pub fn project_c310_pb_rvec_scalar_init(
    slot: &[u8; C310_PB_SLOT_BYTES],
) -> C310PbRvecScalarProjection {
    let big_flags = u32::from_le_bytes(slot[..4].try_into().expect("four-byte flag word"));
    let mut writes = Vec::with_capacity(96);
    let mut payload_cursor = 4_usize;
    for even_register in (0_u8..64).step_by(2) {
        let bit = even_register / 2;
        let source_payload_word = if even_register == 0 || even_register >= 60 {
            None
        } else if big_flags & (1_u32 << bit) != 0 {
            Some(((payload_cursor - 4) / 4) as u8)
        } else {
            continue;
        };
        let (first, second) = if source_payload_word.is_some() {
            let first = u16::from_le_bytes(
                slot[payload_cursor..payload_cursor + 2]
                    .try_into()
                    .expect("two-byte scalar value"),
            );
            let second = u16::from_le_bytes(
                slot[payload_cursor + 2..payload_cursor + 4]
                    .try_into()
                    .expect("two-byte scalar value"),
            );
            payload_cursor += 4;
            (u32::from(first), u32::from(second))
        } else {
            (0, 0)
        };
        writes.extend([
            C310RvecScalarWrite {
                register_index: even_register,
                value: first,
                source_payload_word,
            },
            C310RvecScalarWrite {
                register_index: even_register + 1,
                value: second,
                source_payload_word,
            },
            C310RvecScalarWrite {
                register_index: 64 + bit,
                value: first | (second << 16),
                source_payload_word,
            },
        ]);
    }
    C310PbRvecScalarProjection {
        big_flags,
        consumed_payload_words: ((payload_cursor - 4) / 4) as u8,
        writes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_flags_consume_packed_values_in_bit_order() {
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        slot[..4].copy_from_slice(&((1_u32 << 1) | (1_u32 << 3)).to_le_bytes());
        slot[4..8].copy_from_slice(&0x7654_3210_u32.to_le_bytes());
        slot[8..12].copy_from_slice(&0xfedc_ba98_u32.to_le_bytes());
        let projection = project_c310_pb_rvec_scalar_init(&slot);
        assert_eq!(projection.consumed_payload_words, 2);
        assert_eq!(projection.writes.len(), 15);
        assert_eq!(projection.writes[0].register_index, 0);
        assert_eq!(projection.writes[0].source_payload_word, None);
        assert_eq!(projection.writes[3].register_index, 2);
        assert_eq!(projection.writes[3].value, 0x3210);
        assert_eq!(projection.writes[4].register_index, 3);
        assert_eq!(projection.writes[4].value, 0x7654);
        assert_eq!(projection.writes[5].register_index, 65);
        assert_eq!(projection.writes[5].value, 0x7654_3210);
        assert_eq!(projection.writes[5].source_payload_word, Some(0));
        assert_eq!(projection.writes[6].register_index, 6);
        assert_eq!(projection.writes[6].value, 0xba98);
        assert_eq!(projection.writes[8].register_index, 67);
        assert_eq!(projection.writes[8].value, 0xfedc_ba98);
        assert_eq!(projection.writes[8].source_payload_word, Some(1));
        assert_eq!(projection.writes[9].register_index, 60);
        assert_eq!(projection.writes[12].register_index, 62);
    }

    #[test]
    fn boundary_bits_do_not_consume_payload_words() {
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        slot[..4].copy_from_slice(&((1_u32 << 0) | (1_u32 << 30) | (1_u32 << 31)).to_le_bytes());
        slot[4..8].copy_from_slice(&0xdead_beef_u32.to_le_bytes());
        let projection = project_c310_pb_rvec_scalar_init(&slot);
        assert_eq!(projection.consumed_payload_words, 0);
        assert_eq!(projection.writes.len(), 9);
        assert!(projection.writes.iter().all(|write| write.value == 0));
        assert_eq!(projection.writes[0].register_index, 0);
        assert_eq!(projection.writes[3].register_index, 60);
        assert_eq!(projection.writes[6].register_index, 62);
    }

    #[test]
    fn full_flag_word_stays_within_the_slot() {
        let mut slot = [0_u8; C310_PB_SLOT_BYTES];
        slot[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        for payload_word in 0_u32..29 {
            let offset = 4 + payload_word as usize * 4;
            slot[offset..offset + 4].copy_from_slice(&payload_word.to_le_bytes());
        }
        let projection = project_c310_pb_rvec_scalar_init(&slot);
        assert_eq!(projection.consumed_payload_words, 29);
        assert_eq!(projection.writes.len(), 96);
        assert_eq!(projection.writes[87].register_index, 58);
        assert_eq!(projection.writes[87].source_payload_word, Some(28));
        assert_eq!(projection.writes[90].register_index, 60);
        assert_eq!(projection.writes[90].source_payload_word, None);
    }
}
