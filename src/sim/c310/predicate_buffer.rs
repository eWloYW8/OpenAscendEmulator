use crate::isa::c310::dispatch::C310PushPbStep;
use crate::isa::c310::layout::{C310_PB_DEFAULT_SLOTS, C310_PB_PUSH_BYTES, C310_PB_SLOT_BYTES};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C310PushPbDisposition {
    Accepted,
    Stalled,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum C310PredicateBufferError {
    #[error("predicate-buffer slot count must be positive")]
    NoSlots,
    #[error("predicate-buffer slot {id} is outside configured range 0..{count}")]
    InvalidSlot { id: u16, count: u16 },
    #[error("predicate-buffer slot is full until the owner updates the slot cursor")]
    SlotCursorExhausted,
    #[error("no predicate-buffer slot is active for vector issue")]
    NoActiveSlot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310PredicateBufferWrite {
    pub slot_id: u16,
    pub byte_offset: u8,
    pub bytes: [u8; C310_PB_PUSH_BYTES],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C310PredicateBufferVfIssue {
    pub slot_id: u16,
    pub halfword_cursor_before: u8,
    pub next_available_slot: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C310PredicateBuffer {
    slots: Vec<[u8; C310_PB_SLOT_BYTES]>,
    occupied: Vec<bool>,
    active_slot: Option<u16>,
    halfword_cursor: u8,
}

impl C310PredicateBuffer {
    pub fn new(slot_count: u16) -> Result<Self, C310PredicateBufferError> {
        if slot_count == 0 {
            return Err(C310PredicateBufferError::NoSlots);
        }
        Ok(Self {
            slots: vec![[0; C310_PB_SLOT_BYTES]; usize::from(slot_count)],
            occupied: vec![false; usize::from(slot_count)],
            active_slot: None,
            halfword_cursor: 0,
        })
    }

    pub fn with_default_slots() -> Self {
        Self::new(C310_PB_DEFAULT_SLOTS).expect("default slot count is positive")
    }

    pub fn slot_count(&self) -> u16 {
        self.slots.len() as u16
    }

    pub fn occupied_slots(&self) -> usize {
        self.occupied.iter().filter(|&&occupied| occupied).count()
    }

    pub const fn active_slot(&self) -> Option<u16> {
        self.active_slot
    }

    pub const fn halfword_cursor(&self) -> u8 {
        self.halfword_cursor
    }

    pub fn is_full(&self) -> bool {
        self.occupied.iter().all(|occupied| *occupied)
    }

    pub fn push(
        &mut self,
        step: C310PushPbStep,
    ) -> Result<Option<C310PredicateBufferWrite>, C310PredicateBufferError> {
        if self.is_full() {
            return Ok(None);
        }
        let byte_offset = usize::from(self.halfword_cursor) * 2;
        if byte_offset + C310_PB_PUSH_BYTES > C310_PB_SLOT_BYTES {
            return Err(C310PredicateBufferError::SlotCursorExhausted);
        }
        let slot_id = if self.halfword_cursor == 0 {
            self.occupied
                .iter()
                .position(|occupied| !occupied)
                .expect("a non-full predicate buffer has an idle slot") as u16
        } else {
            self.active_slot
                .expect("a nonzero predicate-buffer cursor has an active slot")
        };
        self.slots[usize::from(slot_id)][byte_offset..byte_offset + C310_PB_PUSH_BYTES]
            .copy_from_slice(&step.bytes);
        if self.halfword_cursor == 0 {
            self.occupied[usize::from(slot_id)] = true;
            self.active_slot = Some(slot_id);
        }
        self.halfword_cursor += (C310_PB_PUSH_BYTES / 2) as u8;
        Ok(Some(C310PredicateBufferWrite {
            slot_id,
            byte_offset: byte_offset as u8,
            bytes: step.bytes,
        }))
    }

    pub fn update_slot(&mut self) -> Option<u16> {
        self.halfword_cursor = 0;
        self.occupied
            .iter()
            .position(|occupied| !occupied)
            .map(|index| index as u16)
    }

    pub fn advance_for_vf_issue(
        &mut self,
    ) -> Result<C310PredicateBufferVfIssue, C310PredicateBufferError> {
        let slot_id = self
            .active_slot
            .ok_or(C310PredicateBufferError::NoActiveSlot)?;
        let halfword_cursor_before = self.halfword_cursor;
        let next_available_slot = self.update_slot();
        Ok(C310PredicateBufferVfIssue {
            slot_id,
            halfword_cursor_before,
            next_available_slot,
        })
    }

    pub fn recycle_slot(&mut self, id: u16) -> Result<(), C310PredicateBufferError> {
        let count = self.slot_count();
        let occupied = self
            .occupied
            .get_mut(usize::from(id))
            .ok_or(C310PredicateBufferError::InvalidSlot { id, count })?;
        *occupied = false;
        Ok(())
    }

    pub fn read_slot(&self, id: u16) -> Result<[u8; C310_PB_SLOT_BYTES], C310PredicateBufferError> {
        self.slots
            .get(usize::from(id))
            .copied()
            .ok_or(C310PredicateBufferError::InvalidSlot {
                id,
                count: self.slot_count(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::isa::c310::dispatch::C310PushPbInstruction;

    fn step(word: u32, xregs: &[u64; 32]) -> C310PushPbStep {
        C310PushPbInstruction::decode(Architecture::Dav3510, word)
            .unwrap()
            .resolve(0x1000, xregs)
    }

    #[test]
    fn vf_issue_uses_active_slot_and_advances_cursor() {
        let mut buffer = C310PredicateBuffer::new(2).unwrap();
        let before = buffer.clone();
        assert_eq!(
            buffer.advance_for_vf_issue(),
            Err(C310PredicateBufferError::NoActiveSlot)
        );
        assert_eq!(buffer, before);

        let step = step(0x4314_b108, &[0; 32]);
        assert_eq!(buffer.push(step).unwrap().unwrap().slot_id, 0);
        assert_eq!(
            buffer.advance_for_vf_issue().unwrap(),
            C310PredicateBufferVfIssue {
                slot_id: 0,
                halfword_cursor_before: 16,
                next_available_slot: Some(1),
            }
        );
        assert_eq!(buffer.active_slot(), Some(0));
        assert_eq!(buffer.halfword_cursor(), 0);
        assert_eq!(buffer.push(step).unwrap().unwrap().slot_id, 1);
        assert_eq!(
            buffer.advance_for_vf_issue().unwrap(),
            C310PredicateBufferVfIssue {
                slot_id: 1,
                halfword_cursor_before: 16,
                next_available_slot: None,
            }
        );
    }

    #[test]
    fn observed_words_decode_four_registers_and_little_endian_payload() {
        let mut xregs = [0; 32];
        xregs[10] = 0x0102_0304_0506_0708;
        xregs[11] = 0x1112_1314_1516_1718;
        xregs[2] = 0x2122_2324_2526_2728;
        let add = step(0x4314_b108, &xregs);
        assert_eq!(add.instruction.source_registers, [10, 11, 2, 2]);
        assert_eq!(
            add.source_values,
            [xregs[10], xregs[11], xregs[2], xregs[2]]
        );
        assert_eq!(&add.bytes[..8], &xregs[10].to_le_bytes());
        assert_eq!(&add.bytes[8..16], &xregs[11].to_le_bytes());
        assert_eq!(&add.bytes[16..24], &xregs[2].to_le_bytes());
        assert_eq!(&add.bytes[24..32], &xregs[2].to_le_bytes());
        assert_eq!(
            step(0x4338_2108, &xregs).instruction.source_registers,
            [28, 2, 2, 2]
        );
        assert_eq!(
            step(0x4319_7108, &xregs).instruction.source_registers,
            [12, 23, 2, 2]
        );
        assert!(C310PushPbInstruction::decode(Architecture::Dav2201, 0x4314_b108).is_none());
        assert!(C310PushPbInstruction::decode(Architecture::Dav3510, 0x4314_b10c).is_none());
    }

    #[test]
    fn four_pushes_fill_a_slot_and_update_selects_the_next_idle_slot() {
        let mut buffer = C310PredicateBuffer::new(2).unwrap();
        let mut xregs = [0; 32];
        for index in 0..4 {
            xregs[10] = index;
            let write = buffer.push(step(0x4314_b108, &xregs)).unwrap().unwrap();
            assert_eq!(write.slot_id, 0);
            assert_eq!(
                write.byte_offset,
                (index as usize * C310_PB_PUSH_BYTES) as u8
            );
        }
        assert_eq!(buffer.halfword_cursor(), 64);
        assert_eq!(buffer.occupied_slots(), 1);
        assert_eq!(buffer.read_slot(0).unwrap()[96..104], 3_u64.to_le_bytes());
        let prior = buffer.clone();
        assert_eq!(
            buffer.push(step(0x4314_b108, &xregs)),
            Err(C310PredicateBufferError::SlotCursorExhausted)
        );
        assert_eq!(buffer, prior);
        assert_eq!(buffer.update_slot(), Some(1));
        assert_eq!(buffer.halfword_cursor(), 0);
        assert_eq!(
            buffer
                .push(step(0x4314_b108, &xregs))
                .unwrap()
                .unwrap()
                .slot_id,
            1
        );
        assert!(buffer.is_full());
        let prior = buffer.clone();
        assert_eq!(buffer.push(step(0x4314_b108, &xregs)).unwrap(), None);
        assert_eq!(buffer, prior);
        assert_eq!(buffer.update_slot(), None);
        buffer.recycle_slot(0).unwrap();
        assert_eq!(buffer.update_slot(), Some(0));
        assert_eq!(
            buffer
                .push(step(0x4314_b108, &xregs))
                .unwrap()
                .unwrap()
                .slot_id,
            0
        );
    }

    #[test]
    fn recycle_preserves_slot_bytes_and_invalid_ids_do_not_mutate() {
        assert_eq!(
            C310PredicateBuffer::new(0),
            Err(C310PredicateBufferError::NoSlots)
        );
        let mut buffer = C310PredicateBuffer::new(2).unwrap();
        let mut xregs = [0; 32];
        xregs[10] = 0xdead_beef;
        buffer.push(step(0x4314_b108, &xregs)).unwrap();
        buffer.recycle_slot(0).unwrap();
        assert_eq!(&buffer.read_slot(0).unwrap()[..8], &xregs[10].to_le_bytes());
        let prior = buffer.clone();
        assert_eq!(
            buffer.recycle_slot(2),
            Err(C310PredicateBufferError::InvalidSlot { id: 2, count: 2 })
        );
        assert_eq!(
            buffer.read_slot(2),
            Err(C310PredicateBufferError::InvalidSlot { id: 2, count: 2 })
        );
        assert_eq!(buffer, prior);
    }

    #[test]
    fn partial_slot_read_and_reuse_preserve_explicit_byte_state() {
        let mut buffer = C310PredicateBuffer::new(1).unwrap();
        let mut xregs = [0; 32];
        xregs[10] = 0x0102_0304_0506_0708;
        xregs[11] = 0x1112_1314_1516_1718;
        xregs[2] = 0x2122_2324_2526_2728;
        let first = step(0x4314_b108, &xregs);
        buffer.push(first).unwrap();
        let initial = buffer.read_slot(0).unwrap();
        assert_eq!(&initial[..32], &first.bytes);
        assert!(initial[32..].iter().all(|byte| *byte == 0));
        assert_eq!(buffer.update_slot(), None);
        buffer.recycle_slot(0).unwrap();
        assert_eq!(buffer.update_slot(), Some(0));
        xregs[10] = 9;
        let second = step(0x4314_b108, &xregs);
        buffer.push(second).unwrap();
        let reused = buffer.read_slot(0).unwrap();
        assert_eq!(&reused[..32], &second.bytes);
        assert!(reused[32..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn explicit_source_values_pack_in_operand_order() {
        let instruction =
            C310PushPbInstruction::decode(Architecture::Dav3510, 0x4338_2108).unwrap();
        assert_eq!(instruction.source_registers, [28, 2, 2, 2]);
        let values = [1_u64, 2, 3, 3];
        let step = instruction.from_source_values(0x1000, values);
        assert_eq!(step.source_values, values);
        for (chunk, value) in step.bytes.chunks_exact(8).zip(values) {
            assert_eq!(chunk, value.to_le_bytes());
        }
    }
}
