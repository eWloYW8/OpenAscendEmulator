pub(super) const fn special_flow_form(word: u32) -> Option<u8> {
    if word >> 29 != 2 || (word >> 27) & 3 == 1 || (word >> 21) & 15 != 15 {
        return None;
    }
    Some(((word >> 18) & 7) as u8)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220SetCrossCoreInstruction {
    pub word: u32,
    pub source_register: u8,
    pub pipe_code: u8,
}

impl C220SetCrossCoreInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        if !matches!(special_flow_form(word), Some(4 | 5)) {
            return None;
        }
        Some(Self {
            word,
            source_register: ((word >> 2) & 31) as u8,
            pipe_code: ((word >> 10) & 15) as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ControlElement {
    Byte,
    Half,
    Word,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220VectorControlOperation {
    ScMerge(C220ControlElement),
    ScSplit(C220ControlElement),
    Padding(C220ControlElement),
    Scatter(C220ControlElement),
    L0cLockSet,
    L0cLockRelease,
    Shuffle(C220ControlElement),
    BlockDescriptor { mode: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220VectorControlInstruction {
    pub operation: C220VectorControlOperation,
}

impl C220VectorControlInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        let operation = if word & 0xffc0_0000 == 0x80c0_0000 {
            C220VectorControlOperation::BlockDescriptor {
                mode: ((word >> 7) & 0xf) as u8,
            }
        } else {
            match word & 0xffc0_007f {
                0x8000_0001 => C220VectorControlOperation::ScMerge(C220ControlElement::Byte),
                0x8000_0009 => C220VectorControlOperation::ScMerge(C220ControlElement::Half),
                0x8000_0002 => C220VectorControlOperation::ScSplit(C220ControlElement::Byte),
                0x8000_000a => C220VectorControlOperation::ScSplit(C220ControlElement::Half),
                0x8000_0022 => C220VectorControlOperation::Padding(C220ControlElement::Half),
                0x8000_002a => C220VectorControlOperation::Padding(C220ControlElement::Word),
                0x8000_0062 => C220VectorControlOperation::Scatter(C220ControlElement::Half),
                0x8000_006a => C220VectorControlOperation::Scatter(C220ControlElement::Word),
                0x8000_0012 => C220VectorControlOperation::L0cLockSet,
                0x8000_001a => C220VectorControlOperation::L0cLockRelease,
                0x8000_0007 => C220VectorControlOperation::Shuffle(C220ControlElement::Byte),
                0x8000_000f => C220VectorControlOperation::Shuffle(C220ControlElement::Half),
                0x8000_0017 => C220VectorControlOperation::Shuffle(C220ControlElement::Word),
                _ => return None,
            }
        };
        Some(Self { operation })
    }
}
