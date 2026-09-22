#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ReductionWidth {
    F16,
    S16,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ExtremumOperation {
    Maximum,
    Minimum,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ExtremumOutput {
    ValueIndex,
    IndexValue,
    Value,
    Index,
}

impl C220ExtremumOutput {
    const fn decode(mode: u32) -> Self {
        match mode & 3 {
            0 => Self::ValueIndex,
            1 => Self::IndexValue,
            2 => Self::Value,
            3 => Self::Index,
            _ => unreachable!(),
        }
    }
}

impl C220ReductionWidth {
    pub const fn element_bytes(self) -> u8 {
        match self {
            Self::F16 | Self::S16 => 2,
            Self::F32 => 4,
        }
    }

    pub const fn lanes(self) -> usize {
        256 / self.element_bytes() as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum C220ReductionKind {
    WholeAdd {
        write_accumulator: bool,
    },
    WholeExtremum {
        operation: C220ExtremumOperation,
        output: C220ExtremumOutput,
    },
    GroupAdd,
    GroupExtremum {
        operation: C220ExtremumOperation,
    },
    PairAdd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220ReductionInstruction {
    pub width: C220ReductionWidth,
    pub kind: C220ReductionKind,
    pub destination_register: u8,
    pub source_register: u8,
    pub control_register: u8,
}

impl C220ReductionInstruction {
    pub const fn decode(word: u32) -> Option<Self> {
        const REGISTERS: u32 = 0x003e_0000 | 0x0001_f000 | 0x0000_007c;
        let add_fixed = word & !(REGISTERS | 0x0000_0002);
        let (width, kind) = match add_fixed {
            0x8340_0380 => (
                C220ReductionWidth::F16,
                C220ReductionKind::WholeAdd {
                    write_accumulator: word & 2 != 0,
                },
            ),
            0x83c0_0380 => (
                C220ReductionWidth::F32,
                C220ReductionKind::WholeAdd {
                    write_accumulator: word & 2 != 0,
                },
            ),
            _ => {
                let whole_fixed = word & !(REGISTERS | 0x0000_0003);
                match whole_fixed {
                    0x8340_0400 => (
                        C220ReductionWidth::F16,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Maximum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    0x83c0_0400 => (
                        C220ReductionWidth::F32,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Maximum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    0x8380_0400 => (
                        C220ReductionWidth::S16,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Maximum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    0x8340_0480 => (
                        C220ReductionWidth::F16,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Minimum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    0x83c0_0480 => (
                        C220ReductionWidth::F32,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Minimum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    0x8380_0480 => (
                        C220ReductionWidth::S16,
                        C220ReductionKind::WholeExtremum {
                            operation: C220ExtremumOperation::Minimum,
                            output: C220ExtremumOutput::decode(word),
                        },
                    ),
                    _ => {
                        let fixed = word & !REGISTERS;
                        match fixed {
                            0x8340_0500 => (
                                C220ReductionWidth::F16,
                                C220ReductionKind::GroupExtremum {
                                    operation: C220ExtremumOperation::Maximum,
                                },
                            ),
                            0x83c0_0500 => (
                                C220ReductionWidth::F32,
                                C220ReductionKind::GroupExtremum {
                                    operation: C220ExtremumOperation::Maximum,
                                },
                            ),
                            0x8340_0580 => (
                                C220ReductionWidth::F16,
                                C220ReductionKind::GroupExtremum {
                                    operation: C220ExtremumOperation::Minimum,
                                },
                            ),
                            0x83c0_0580 => (
                                C220ReductionWidth::F32,
                                C220ReductionKind::GroupExtremum {
                                    operation: C220ExtremumOperation::Minimum,
                                },
                            ),
                            0x8340_0600 => (C220ReductionWidth::F16, C220ReductionKind::GroupAdd),
                            0x83c0_0600 => (C220ReductionWidth::F32, C220ReductionKind::GroupAdd),
                            0x8340_0780 => (C220ReductionWidth::F16, C220ReductionKind::PairAdd),
                            0x83c0_0780 => (C220ReductionWidth::F32, C220ReductionKind::PairAdd),
                            _ => return None,
                        }
                    }
                }
            }
        };
        Some(Self {
            width,
            kind,
            destination_register: ((word >> 17) & 0x1f) as u8,
            source_register: ((word >> 12) & 0x1f) as u8,
            control_register: ((word >> 2) & 0x1f) as u8,
        })
    }

    pub const fn writes_accumulator(self) -> bool {
        matches!(
            self.kind,
            C220ReductionKind::WholeAdd {
                write_accumulator: true
            }
        )
    }

    pub const fn has_cross_repeat_state(self) -> bool {
        self.writes_accumulator() || matches!(self.kind, C220ReductionKind::WholeExtremum { .. })
    }
}
