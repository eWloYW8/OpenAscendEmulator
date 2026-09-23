use super::super::{C220FixpCommand, C220FixpOutputFormat};
use super::{
    C220FixpNz2ndReadError, C220FixpNz2ndReadGenerator, C220FixpNz2ndWriteError,
    C220FixpNz2ndWritePlanner, C220FixpNz2ndWriteUop,
};

#[derive(Debug, thiserror::Error)]
pub enum C220FixpNz2ndPlanError {
    #[error(transparent)]
    Read(#[from] C220FixpNz2ndReadError),
    #[error(transparent)]
    Write(#[from] C220FixpNz2ndWriteError),
}

/// Read and transpose-write streams from one captured command. Each stream
/// expands independently, without materializing the full coordinate matrix.
#[derive(Debug, Clone)]
pub struct C220FixpNz2ndInstructionPlan {
    pub reads: C220FixpNz2ndReadGenerator,
    pub writes: C220FixpNz2ndWriteGenerator,
}

impl C220FixpNz2ndInstructionPlan {
    pub fn new(
        command: C220FixpCommand,
        instruction_id: u64,
        first_read_id: u32,
        first_write_id: u32,
        bandwidth: u32,
        main_slots: u32,
    ) -> Result<Self, C220FixpNz2ndPlanError> {
        let reads =
            C220FixpNz2ndReadGenerator::new(command, instruction_id, first_read_id, bandwidth)?;
        let planner = C220FixpNz2ndWritePlanner::new(command, main_slots)?;
        let lane_bytes = C220FixpOutputFormat::from_conversion_mode(
            command.source_format,
            command.descriptor.conversion_mode(),
        )
        .expect("validated command")
        .lane_bytes();
        Ok(Self {
            writes: C220FixpNz2ndWriteGenerator {
                command,
                instruction_id,
                next_id: first_write_id,
                reads: reads.clone(),
                planner,
                main_slots,
                lane_bytes,
                position: None,
                group_index: 0,
                destination_column: 0,
                advance_column: None,
                rows: None,
                tail: None,
            },
            reads,
        })
    }
}

#[derive(Debug, Clone)]
struct RowGroup {
    nd: u32,
    row: u32,
    end_row: u32,
    column: u32,
    group_index: u32,
    end_of_group: bool,
    end_of_nd: bool,
}

#[derive(Debug, Clone)]
pub struct C220FixpNz2ndWriteGenerator {
    command: C220FixpCommand,
    instruction_id: u64,
    next_id: u32,
    reads: C220FixpNz2ndReadGenerator,
    planner: C220FixpNz2ndWritePlanner,
    main_slots: u32,
    lane_bytes: u32,
    position: Option<(u32, u32, u32)>,
    group_index: u32,
    destination_column: u32,
    advance_column: Option<u32>,
    rows: Option<RowGroup>,
    tail: Option<C220FixpNz2ndWriteUop>,
}

impl Iterator for C220FixpNz2ndWriteGenerator {
    type Item = C220FixpNz2ndWriteUop;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tail) = self.tail.take() {
            return Some(tail);
        }
        loop {
            if let Some(rows) = self.rows.as_mut() {
                let d = self.command.descriptor;
                let destination = self
                    .command
                    .destination_address
                    .wrapping_add(u64::from(
                        rows.nd
                            .wrapping_mul(d.destination_nd_stride())
                            .wrapping_mul(self.lane_bytes),
                    ))
                    .wrapping_add(u64::from(
                        rows.row
                            .wrapping_mul(d.destination_stride())
                            .wrapping_mul(self.lane_bytes),
                    ))
                    .wrapping_add(u64::from(rows.column) * 16 * u64::from(self.lane_bytes));
                let mut batch =
                    self.planner
                        .plan_row(rows.group_index, rows.end_of_group, destination);
                rows.row += 1;
                let final_row = rows.row == rows.end_row;
                if final_row && rows.end_of_nd {
                    batch.tail.end_of_burst = true;
                }
                let last_in_instruction =
                    final_row && rows.end_of_nd && rows.nd + 1 == u32::from(d.nd_count());
                let first_id = self.next_id;
                self.next_id = self
                    .next_id
                    .wrapping_add(1 + u32::from(batch.leading.is_some()));
                let tail = C220FixpNz2ndWriteUop {
                    instruction_id: self.instruction_id,
                    request_id: first_id.wrapping_add(u32::from(batch.leading.is_some())),
                    descriptor: batch.tail,
                    last_in_instruction,
                };
                if final_row {
                    self.rows = None;
                }
                if let Some(leading) = batch.leading {
                    self.tail = Some(tail);
                    return Some(C220FixpNz2ndWriteUop {
                        instruction_id: self.instruction_id,
                        request_id: first_id,
                        descriptor: leading,
                        last_in_instruction: false,
                    });
                }
                return Some(tail);
            }
            let position @ (nd, row, column) = self.reads.position();
            let read = self.reads.next()?.operation;
            if self
                .position
                .is_none_or(|(old_nd, old_row, _)| (old_nd, old_row) != (nd, row))
            {
                self.group_index = 0;
                self.destination_column = 0;
                self.advance_column = None;
            } else if self
                .position
                .is_some_and(|(_, _, old_column)| old_column != column)
                && let Some(next_column) = self.advance_column.take()
            {
                self.destination_column = next_column;
            }
            self.position = Some(position);
            if !read.last_in_uop {
                continue;
            }
            let group_index = self.group_index;
            self.group_index += 1;
            if group_index % self.main_slots != self.main_slots - 1 && !read.end_of_burst {
                continue;
            }
            let end_row = row + read.data_bytes / 64;
            self.rows = Some(RowGroup {
                nd,
                row,
                end_row,
                column: self.destination_column,
                group_index,
                end_of_group: read.end_of_burst,
                end_of_nd: read.end_of_burst
                    && end_row == u32::from(self.command.descriptor.rows()),
            });
            if !self.planner.gather() {
                self.advance_column = Some(column + 1);
            }
        }
    }
}

impl std::iter::FusedIterator for C220FixpNz2ndWriteGenerator {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::mte::fixp::C220FixpSourceFormat;

    #[test]
    fn instruction_streams_close_each_nd_and_keep_leading_descriptors_row_major() {
        let mut command = C220FixpCommand {
            descriptor: C220FixpDescriptor {
                xt: (24 << 32) | (5 << 16) | (24 << 4),
                xm: (1 << 43) | 16,
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
        let plan = C220FixpNz2ndInstructionPlan::new(command, 9, 0, 100, 256, 8).unwrap();
        assert_eq!(plan.reads.count(), 12);
        let writes: Vec<_> = plan.writes.collect();
        assert_eq!(writes.len(), 10);
        for (i, write) in writes.iter().enumerate() {
            assert_eq!(
                write.descriptor.destination_address,
                4096 + (i / 5) as u64 * 1024 + (i % 5) as u64 * 96
            );
            assert_eq!(write.descriptor.bytes, 96);
            assert!(write.descriptor.gather);
            assert_eq!(write.descriptor.end_of_burst, i % 5 == 4);
            assert_eq!(write.last_in_instruction, i == 9);
            assert_eq!(write.request_id, 100 + i as u32);
        }
        command.descriptor.xt = (160 << 32) | (2 << 16) | (145 << 4);
        command.descriptor.xm |= 1 << 34;
        command.descriptor.nd = 1;
        let writes: Vec<_> = C220FixpNz2ndInstructionPlan::new(command, 9, 0, 0, 256, 8)
            .unwrap()
            .writes
            .collect();
        assert_eq!(
            writes
                .iter()
                .map(|w| (
                    w.descriptor.destination_address,
                    w.descriptor.bytes,
                    w.descriptor.last_in_uop
                ))
                .collect::<Vec<_>>(),
            [
                (4096, 128, false),
                (4096, 256, true),
                (4416, 128, false),
                (4416, 256, true),
                (4352, 34, true),
                (4672, 34, true)
            ]
        );
        assert_eq!(writes.iter().filter(|w| w.last_in_instruction).count(), 1);
        assert!(writes.last().unwrap().last_in_instruction);
    }
}
