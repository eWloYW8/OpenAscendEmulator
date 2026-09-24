use super::super::{C220FixpCommand, C220FixpLayoutError, C220FixpReadUop};
use crate::sim::c220::cube::C220CubeL0cAccess;
use crate::sim::c220::memory::{C220L0cFragmentRequest, C220L0cReadRequest};
use crate::sim::c220::mte::interface::C220MteL0cReadOperation;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum C220FixpNz2ndReadError {
    #[error(transparent)]
    Layout(#[from] C220FixpLayoutError),
    #[error("NZ2ND read generation requires an NZ-to-ND command")]
    NotNzToNd,
    #[error("NZ2ND read bandwidth must be a nonzero multiple of 64 bytes")]
    Bandwidth,
    #[error("NZ2ND source address must align to a physical 16-lane row")]
    SourceAlignment,
    #[error(
        "NZ2ND read cannot complete whole timing rows at ND {nd_index}, row {row}: {data_bytes} bytes"
    )]
    IncompleteTimingRow {
        nd_index: u32,
        row: u32,
        data_bytes: u32,
    },
}

/// Lazy row-block traversal. Every column in a block uses the byte count
/// selected at the first column; 32-bit output repeats each full-column read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpNz2ndReadGenerator {
    command: C220FixpCommand,
    instruction_id: u64,
    next_id: u32,
    bandwidth: u32,
    nd: u32,
    nd_count: u32,
    row: u32,
    column: u32,
    columns: u32,
    part: u32,
}

impl C220FixpNz2ndReadGenerator {
    pub(super) fn position(&self) -> (u32, u32, u32) {
        (self.nd, self.row, self.column)
    }

    pub fn new(
        command: C220FixpCommand,
        instruction_id: u64,
        first_id: u32,
        bandwidth: u32,
    ) -> Result<Self, C220FixpNz2ndReadError> {
        command.layout()?;
        if !command.descriptor.nz_to_nd() {
            return Err(C220FixpNz2ndReadError::NotNzToNd);
        }
        if bandwidth == 0 || !bandwidth.is_multiple_of(64) {
            return Err(C220FixpNz2ndReadError::Bandwidth);
        }
        if !command
            .source_address
            .is_multiple_of(u64::from(16 * command.source_format.lane_bytes()))
        {
            return Err(C220FixpNz2ndReadError::SourceAlignment);
        }
        let generator = Self {
            command,
            instruction_id,
            next_id: first_id,
            bandwidth,
            nd: 0,
            nd_count: if command.descriptor.is_disabled() {
                0
            } else {
                u32::from(command.descriptor.nd_count())
            },
            row: 0,
            column: 0,
            columns: u32::from(command.descriptor.columns()).div_ceil(16),
            part: 0,
        };
        generator.validate_row_progress()?;
        Ok(generator)
    }

    fn source_address(&self, column: u32) -> u64 {
        self.source_address_at(self.nd, self.row, column)
    }

    fn source_address_at(&self, nd: u32, row: u32, column: u32) -> u64 {
        let d = self.command.descriptor;
        let source_row_bytes = 16 * self.command.source_format.lane_bytes();
        self.command
            .source_address
            .wrapping_add(u64::from(
                nd.wrapping_mul(16 * source_row_bytes)
                    .wrapping_mul(u32::from(d.source_nd_stride())),
            ))
            .wrapping_add(u64::from(row) * u64::from(source_row_bytes))
            .wrapping_add(u64::from(
                column
                    .wrapping_mul(source_row_bytes)
                    .wrapping_mul(u32::from(d.source_stride())),
            ))
    }

    fn validate_row_progress(&self) -> Result<(), C220FixpNz2ndReadError> {
        if self.command.source_format.lane_bytes() == 4 {
            return Ok(());
        }
        // Source addresses use physical lane widths, but timing consumes
        // 64-byte rows. A partial timing row loses byte-count progress without
        // advancing the corresponding row count and cannot finish the command.
        for nd_index in 0..self.nd_count {
            let mut row = 0;
            while row < u32::from(self.command.descriptor.rows()) {
                let address = self.source_address_at(nd_index, row, 0);
                let boundary = self.bandwidth - address as u32 % self.bandwidth;
                let data_bytes =
                    ((u32::from(self.command.descriptor.rows()) - row) * 64).min(boundary);
                if !data_bytes.is_multiple_of(64) {
                    return Err(C220FixpNz2ndReadError::IncompleteTimingRow {
                        nd_index,
                        row,
                        data_bytes,
                    });
                }
                row += data_bytes / 64;
            }
        }
        Ok(())
    }
}

impl Iterator for C220FixpNz2ndReadGenerator {
    type Item = C220FixpReadUop;

    fn next(&mut self) -> Option<Self::Item> {
        if self.nd == self.nd_count {
            return None;
        }
        let d = self.command.descriptor;
        let boundary = self.bandwidth - self.source_address(0) as u32 % self.bandwidth;
        let data_bytes = ((u32::from(d.rows()) - self.row) * 64).min(boundary);
        let rows = data_bytes / 64;
        let parts = if d.conversion_mode() != 0 {
            1
        } else if self.column + 1 == self.columns && !d.columns().is_multiple_of(16) {
            u32::from(d.columns() % 16).div_ceil(8)
        } else {
            2
        };
        let address = self.source_address(self.column);
        let begins_unit = self.part == 0 && (self.row == 0 || address.is_multiple_of(1024));
        let end_of_burst = self.column + 1 == self.columns && self.part + 1 == parts;
        let operation = C220MteL0cReadOperation {
            instruction_id: self.instruction_id,
            uop_id: self.next_id,
            begins_unit,
            conversion_mode: u32::from(d.conversion_mode()),
            last_in_instruction: end_of_burst
                && self.row + rows == u32::from(d.rows())
                && self.nd + 1 == self.nd_count,
            request: C220L0cReadRequest {
                id: self.next_id,
                fragments: C220L0cFragmentRequest {
                    address,
                    bytes: 1024,
                    access: C220CubeL0cAccess::Read,
                    check_unit_flags: begins_unit && matches!(d.unit_flag_mode(), 2 | 3),
                    update_unit_flags: begins_unit && d.unit_flag_mode() == 3,
                },
                data_type: 0,
                half_accumulator: false,
            },
            data_bytes,
            destination_address: 0,
            output_bytes: 32 * rows,
            second_channel_offset: None,
            last_in_uop: match d.conversion_mode() {
                8 | 9 | 23 | 24 => self.column % 2 == 1 || self.column + 1 == self.columns,
                21 | 22 | 25 | 26 => self.column % 4 == 3 || self.column + 1 == self.columns,
                _ => true,
            },
            end_of_burst,
        };
        self.next_id = self.next_id.wrapping_add(1);
        self.part += 1;
        if self.part == parts {
            self.part = 0;
            self.column += 1;
            if self.column == self.columns {
                self.column = 0;
                self.row += rows;
                if self.row == u32::from(d.rows()) {
                    self.row = 0;
                    self.nd += 1;
                }
            }
        }
        Some(C220FixpReadUop { operation })
    }
}

impl std::iter::FusedIterator for C220FixpNz2ndReadGenerator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::mte::fixp::{
        C220FixpReadPipeline, C220FixpReadProgress, C220FixpSourceFormat,
    };

    #[test]
    fn row_blocks_duplicate_full_reads_and_preserve_nd_boundaries_in_pipeline() {
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (32 << 32) | (5 << 16) | (24 << 4),
                xm: (1 << 43) | (3 << 32) | 16,
                nd: (256 << 32) | (2 << 16) | 2,
            },
            source_format: C220FixpSourceFormat::Fp32,
            source_address: 64,
            destination_address: 4096,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
            dequant_base_block: 0,
            scalar_dequant: 0,
        };
        let generator = C220FixpNz2ndReadGenerator::new(command, 7, u32::MAX, 256).unwrap();
        let packets: Vec<_> = generator.clone().map(|p| p.operation).collect();
        assert_eq!(packets.len(), 12);
        assert_eq!(
            packets
                .iter()
                .map(|p| p.request.fragments.address)
                .collect::<Vec<_>>(),
            [
                64, 64, 1088, 256, 256, 1280, 2112, 2112, 3136, 2304, 2304, 3328
            ]
        );
        assert_eq!(
            packets.iter().map(|p| p.data_bytes).collect::<Vec<_>>(),
            [192, 192, 192, 128, 128, 128, 192, 192, 192, 128, 128, 128]
        );
        assert_eq!(
            packets.iter().map(|p| p.output_bytes).collect::<Vec<_>>(),
            [96, 96, 96, 64, 64, 64, 96, 96, 96, 64, 64, 64]
        );
        assert_eq!(
            packets
                .iter()
                .enumerate()
                .filter(|(_, p)| p.begins_unit)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            [0, 2, 6, 8]
        );
        assert_eq!(
            packets
                .iter()
                .enumerate()
                .filter(|(_, p)| p.end_of_burst)
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            [2, 5, 8, 11]
        );
        assert_eq!(packets.iter().filter(|p| p.last_in_instruction).count(), 1);
        assert!(packets[11].last_in_instruction);
        assert_eq!(packets[1].uop_id, 0);
        assert!(!packets[1].request.fragments.update_unit_flags);
        let mut pipeline = C220FixpReadPipeline::default();
        assert!(pipeline.submit(10, generator).unwrap());
        assert_eq!(
            pipeline.generate(11).unwrap(),
            C220FixpReadProgress::Delayed { ready_tick: 12 }
        );
        for tick in 12..19 {
            let C220FixpReadProgress::Advanced(uop) = pipeline.generate(tick).unwrap() else {
                panic!("expected generated read")
            };
            assert_eq!(uop.operation, packets[(tick - 12) as usize]);
        }
        assert_eq!(
            pipeline.generate(19).unwrap(),
            C220FixpReadProgress::QueueFull
        );
        command.descriptor.xm |= 1 << 34;
        assert_eq!(
            C220FixpNz2ndReadGenerator::new(command, 7, 0, 256)
                .unwrap()
                .count(),
            8
        );
        command.descriptor.nd = 0;
        assert_eq!(
            C220FixpNz2ndReadGenerator::new(command, 7, 0, 256)
                .unwrap()
                .count(),
            0
        );
        assert!(matches!(
            C220FixpNz2ndReadGenerator::new(command, 7, 0, 65),
            Err(C220FixpNz2ndReadError::Bandwidth)
        ));
        command.source_format = C220FixpSourceFormat::Fp16;
        command.source_address = 0;
        command.descriptor.xt = (32 << 32) | (3 << 16) | (24 << 4);
        command.descriptor.xm = (1 << 43) | (6 << 34) | 16;
        command.descriptor.nd = (2 << 16) | 2;
        let half: Vec<_> = C220FixpNz2ndReadGenerator::new(command, 7, 0, 128)
            .unwrap()
            .map(|uop| {
                let op = uop.operation;
                (op.request.fragments.address, op.data_bytes, op.output_bytes)
            })
            .collect();
        assert_eq!(
            half,
            [
                (0, 128, 64),
                (512, 128, 64),
                (64, 64, 32),
                (576, 64, 32),
                (1024, 128, 64),
                (1536, 128, 64),
                (1088, 64, 32),
                (1600, 64, 32),
            ]
        );
        command.descriptor.xt = (32 << 32) | (4 << 16) | (24 << 4);
        assert!(matches!(
            C220FixpNz2ndReadGenerator::new(command, 7, 0, 128),
            Err(C220FixpNz2ndReadError::IncompleteTimingRow {
                nd_index: 0,
                row: 3,
                data_bytes: 32,
            })
        ));
        command.source_address = 32;
        command.descriptor.xt = (32 << 32) | (1 << 16) | (16 << 4);
        let first = C220FixpNz2ndReadGenerator::new(command, 7, 0, 128)
            .unwrap()
            .next()
            .unwrap()
            .operation;
        assert_eq!(
            (
                first.request.fragments.address,
                first.data_bytes,
                first.output_bytes
            ),
            (32, 64, 32)
        );
    }
}
