#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AicClass {
    Scalar,
    FlowControl,
    MemoryTransfer,
    Vector,
    Fixp,
    Cube,
    Unregistered(u8),
}

impl AicClass {
    pub const fn from_word(word: u32) -> Self {
        match (word >> 29) as u8 {
            0 => Self::Scalar,
            2 => Self::FlowControl,
            3 => Self::MemoryTransfer,
            4 => Self::Vector,
            6 => Self::Fixp,
            7 => Self::Cube,
            other => Self::Unregistered(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_key_selects_instruction_class() {
        let expected = [
            AicClass::Scalar,
            AicClass::Unregistered(1),
            AicClass::FlowControl,
            AicClass::MemoryTransfer,
            AicClass::Vector,
            AicClass::Unregistered(5),
            AicClass::Fixp,
            AicClass::Cube,
        ];
        for (key, class) in expected.into_iter().enumerate() {
            assert_eq!(AicClass::from_word((key as u32) << 29), class);
        }
    }
}
