use crate::isa::c220::mte::load3d::{
    C220Load3dElement, C220Load3dGeometry, C220Load3dMatrix, C220Load3dRepeat,
    C220Load3dV2Instruction, C220Load3dV2Operands,
};

mod tiles;
pub use tiles::{C220Load3dTile, C220Load3dTiles};
mod coordinates;
pub use coordinates::{C220Load3dCoordinate, C220Load3dCoordinateError, C220Load3dCoordinates};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("LOAD3Dv2 requires SPR {register}")]
pub struct C220Load3dCaptureError {
    pub register: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dDisabledReasons {
    pub zero_repeat: bool,
    pub empty_matrix: bool,
    pub empty_filter: bool,
    pub zero_channels: bool,
    pub empty_extent: bool,
}

impl C220Load3dDisabledReasons {
    pub const fn any(self) -> bool {
        self.zero_repeat
            || self.empty_matrix
            || self.empty_filter
            || self.zero_channels
            || self.empty_extent
    }
}

/// Captured command inputs, before coordinate expansion and physical scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dV2Command {
    pub operands: C220Load3dV2Operands,
    pub matrix: C220Load3dMatrix,
    pub repeat: C220Load3dRepeat,
    pub padding: u64,
    pub l1_size: u64,
    pub disabled: C220Load3dDisabledReasons,
    pub odd_packed_channels: bool,
}

impl C220Load3dV2Command {
    pub fn tiles(self) -> C220Load3dTiles {
        C220Load3dTiles::new(self)
    }

    pub fn coordinates(
        self,
        tile: C220Load3dTile,
    ) -> Result<C220Load3dCoordinates, C220Load3dCoordinateError> {
        C220Load3dCoordinates::new(self, tile)
    }

    pub fn capture(
        instruction: C220Load3dV2Instruction,
        registers: &[u64; 32],
        mut spr: impl FnMut(u16) -> Option<u64>,
    ) -> Result<Self, C220Load3dCaptureError> {
        let operands = instruction.capture(registers);
        let mut read = |register| spr(register).ok_or(C220Load3dCaptureError { register });
        let primary = read(10)?;
        let alternate = read(92)?;
        let padding = read(13)?;
        let repeat = C220Load3dRepeat::decode(read(58)?);
        let l1_size = read(22)?;
        let matrix = C220Load3dMatrix::decode(if operands.geometry.alternate_matrix {
            alternate
        } else {
            primary
        });
        let disabled = C220Load3dDisabledReasons {
            zero_repeat: repeat.count == 0,
            empty_matrix: matrix.width == 0 || matrix.height == 0,
            empty_filter: operands.geometry.filter_w == 0 || operands.geometry.filter_h == 0,
            zero_channels: operands.geometry.channel_size == 0,
            empty_extent: operands.extent.k_length == 0 || operands.extent.m_length == 0,
        };
        Ok(Self {
            operands,
            matrix,
            repeat,
            padding,
            l1_size,
            disabled,
            odd_packed_channels: instruction.element == C220Load3dElement::B4
                && operands.geometry.channel_size & 1 != 0,
        })
    }

    pub fn effective_geometry(self) -> C220Load3dGeometry {
        let mut geometry = self.operands.geometry;
        geometry.stride_w = geometry.stride_w.max(1);
        geometry.stride_h = geometry.stride_h.max(1);
        geometry.filter_w = geometry.filter_w.max(1);
        geometry.filter_h = geometry.filter_h.max(1);
        geometry.dilation_w = geometry.dilation_w.max(1);
        geometry.dilation_h = geometry.dilation_h.max(1);
        geometry
    }

    pub const fn transposed(self) -> bool {
        matches!(
            self.operands.instruction.element,
            C220Load3dElement::B16 | C220Load3dElement::B32
        ) && matches!(
            self.operands.instruction.destination,
            crate::isa::c220::mte::load3d::C220Load3dDestination::L0a
        ) && self.operands.geometry.transpose
    }

    pub const fn effective_l1_size(self) -> u64 {
        if self.l1_size == 0 {
            1 << 20
        } else {
            self.l1_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::load3d::C220Load3dDestination;

    #[test]
    fn captures_encoded_geometry_without_losing_disabled_inputs() {
        for opcode in 0..32 {
            let word = (3 << 29) | (opcode << 22) | (1 << 17) | (2 << 12) | (3 << 7) | (4 << 2);
            let instruction = C220Load3dV2Instruction::decode(word);
            assert_eq!(instruction.is_some(), matches!(opcode, 20..=23 | 28..=31));
            let Some(instruction) = instruction else {
                continue;
            };
            assert_eq!(
                crate::isa::c220::mte::read_register_mask(word),
                Some(0b11110)
            );
            assert!(C220Load3dV2Instruction::decode(word | (1 << 27)).is_none());
            assert_eq!(
                instruction.destination,
                if opcode & 2 != 0 {
                    C220Load3dDestination::L0b
                } else {
                    C220Load3dDestination::L0a
                }
            );
            assert_eq!(
                C220Load3dV2Instruction::decode(word | 2)
                    .unwrap()
                    .destination,
                C220Load3dDestination::L0b
            );
            assert_eq!(
                instruction.element,
                match ((opcode >> 3) & 1, opcode & 1) {
                    (0, 0) => C220Load3dElement::B8,
                    (0, 1) => C220Load3dElement::B16,
                    (1, 0) => C220Load3dElement::B4,
                    _ => C220Load3dElement::B32,
                }
            );
            let mut registers = [0; 32];
            registers[1] = 1024;
            registers[2] = 2048;
            registers[3] = 17 | (33 << 16) | (5 << 32) | (7 << 48);
            registers[4] = (1 << 12)
                | (2 << 20)
                | (1 << 44)
                | (1 << 45)
                | (1 << 46)
                | (1 << 47)
                | (0x1235 << 48);
            let spr = |register| {
                Some(match register {
                    10 => 8 | (9 << 16),
                    92 => 4 | (5 << 16) | (1 << 32) | (2 << 40) | (3 << 48) | (4 << 56),
                    58 => 3 | (2 << 16) | (1 << 24),
                    13 => 0xabcd,
                    22 => 0,
                    _ => return None,
                })
            };
            let command = C220Load3dV2Command::capture(instruction, &registers, spr).unwrap();
            assert_eq!(command.operands.source_base, 2048);
            let extent = command.operands.extent;
            assert_eq!(
                (
                    extent.k_length,
                    extent.m_length,
                    extent.k_start,
                    extent.m_start
                ),
                (17, 33, 5, 7)
            );
            assert_eq!((command.matrix.width, command.matrix.height), (4, 5));
            assert_eq!(
                (
                    command.matrix.pad_left,
                    command.matrix.pad_right,
                    command.matrix.pad_top,
                    command.matrix.pad_bottom
                ),
                (1, 2, 3, 4)
            );
            let geometry = command.operands.geometry;
            assert_eq!(
                (geometry.filter_w, geometry.filter_h, geometry.channel_size),
                (257, 258, 0x1235)
            );
            assert_eq!((geometry.stride_w, geometry.dilation_h), (0, 0));
            assert_eq!(
                (
                    command.effective_geometry().stride_w,
                    command.effective_geometry().dilation_h
                ),
                (1, 1)
            );
            assert_eq!(command.effective_geometry().raw, registers[4]);
            assert_eq!(
                (
                    command.repeat.stride,
                    command.repeat.count,
                    command.repeat.k_mode
                ),
                (3, 2, true)
            );
            assert_eq!(command.effective_l1_size(), 1 << 20);
            assert!(!command.disabled.any());
            assert_eq!(
                command.odd_packed_channels,
                instruction.element == C220Load3dElement::B4
            );
            assert_eq!(
                command.transposed(),
                instruction.destination == C220Load3dDestination::L0a
                    && matches!(
                        instruction.element,
                        C220Load3dElement::B16 | C220Load3dElement::B32
                    )
            );
            let disabled =
                C220Load3dV2Command::capture(instruction, &[0; 32], |_| Some(0)).unwrap();
            assert!(
                disabled.disabled.zero_repeat
                    && disabled.disabled.empty_filter
                    && disabled.disabled.empty_matrix
                    && disabled.disabled.zero_channels
                    && disabled.disabled.empty_extent
            );
            assert_eq!(
                C220Load3dV2Command::capture(instruction, &registers, |r| if r == 92 {
                    None
                } else {
                    spr(r)
                }),
                Err(C220Load3dCaptureError { register: 92 })
            );
        }
    }
}
