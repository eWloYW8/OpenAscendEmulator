use super::{C220FixpCommand, C220FixpExecutionError, C220FixpSliceResult};
use crate::sim::c220::memory::C220LocalBuffer;
use crate::sim::c220::mte::interface::C220MteL0cReadAcknowledgment;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpFunctionalEvent {
    pub tick: u64,
    pub instruction_id: u64,
    pub uop_id: u32,
    pub snapshot_address: Option<u64>,
    pub executed: bool,
}

/// Functional L0C view shared by the FIX engine. It persists across commands;
/// only marked read acceptances update it. Numerical destination writes occur
/// on the final read acceptance, independently of later timing retirement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct C220FixpFunctionalState {
    snapshot: C220LocalBuffer,
}

impl C220FixpFunctionalState {
    pub fn new(l0c_capacity: u64) -> Self {
        Self {
            snapshot: C220LocalBuffer::new(l0c_capacity),
        }
    }

    pub fn snapshot(&self) -> &C220LocalBuffer {
        &self.snapshot
    }

    /// Call exactly once for each accepted L0C response, using its owning
    /// command. Do not call when a read is blocked, at delayed conversion
    /// delivery, or again when the write interface retires the instruction.
    pub fn accept_read(
        &mut self,
        acknowledgment: C220MteL0cReadAcknowledgment,
        command: C220FixpCommand,
        l0c: &C220LocalBuffer,
        slopes: &C220LocalBuffer,
        l1: &mut C220LocalBuffer,
        observe: impl FnMut(&C220FixpSliceResult),
    ) -> Result<C220FixpFunctionalEvent, C220FixpExecutionError> {
        self.accept_read_with(acknowledgment, l0c, |snapshot| {
            command.execute_to_l1(snapshot, slopes, l1, observe)
        })
    }

    /// Updates the shared snapshot and invokes the destination-specific
    /// executor only on the final accepted read. The callback must not advance
    /// timing state; conversion delivery and write retirement are separate.
    pub fn accept_read_with(
        &mut self,
        acknowledgment: C220MteL0cReadAcknowledgment,
        l0c: &C220LocalBuffer,
        execute: impl FnOnce(&C220LocalBuffer) -> Result<(), C220FixpExecutionError>,
    ) -> Result<C220FixpFunctionalEvent, C220FixpExecutionError> {
        let operation = acknowledgment.operation;
        let snapshot_address = operation
            .begins_unit
            .then_some(operation.request.fragments.address);
        if let Some(address) = snapshot_address {
            let bytes = l0c.read_initialized_linear(address, 1024)?;
            self.snapshot.write_known_linear(address, &bytes)?;
        }
        if operation.last_in_instruction {
            execute(&self.snapshot)?;
        }
        Ok(C220FixpFunctionalEvent {
            tick: acknowledgment.accepted_tick,
            instruction_id: operation.instruction_id,
            uop_id: operation.uop_id,
            snapshot_address,
            executed: operation.last_in_instruction,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::c220::mte::fixp::C220FixpDescriptor;
    use crate::sim::c220::mte::fixp::C220FixpReadGenerator;

    #[test]
    fn final_read_uses_unit_snapshot_instead_of_later_live_l0c_values() {
        let command = C220FixpCommand {
            source_format: crate::sim::c220::mte::fixp::C220FixpSourceFormat::Fp32,
            descriptor: C220FixpDescriptor {
                xt: (8 << 16) | (16 << 4),
                xm: 1 << 34,
                nd: 0,
            },
            source_address: 128,
            destination_address: 0,
            control: 0,
            scalar_slope: 0,
            slope_base_block: 0,
            dequant_base_block: 0,
            scalar_dequant: 0,
        };
        let mut live = C220LocalBuffer::new(4096);
        live.write_known_linear(128, &1_f32.to_le_bytes().repeat(256))
            .unwrap();
        let slopes = C220LocalBuffer::new(4096);
        let mut l1 = C220LocalBuffer::new(4096);
        let mut functional = C220FixpFunctionalState::new(4096);
        let mut packets = C220FixpReadGenerator::new(command, 7, 1, 256)
            .unwrap()
            .peekable();
        let mut tick = 10;
        while let Some(packet) = packets.next() {
            let acknowledgment = C220MteL0cReadAcknowledgment {
                operation: packet.operation,
                accepted_tick: tick,
                data_ready_tick: tick + 8,
                queue_ready_tick: tick + 1,
                retry_ready_tick: None,
            };
            let event = functional
                .accept_read(acknowledgment, command, &live, &slopes, &mut l1, |_| {})
                .unwrap();
            assert_eq!(event.executed, packets.peek().is_none());
            if tick == 10 {
                assert_eq!(event.snapshot_address, Some(128));
                assert_eq!(
                    functional.snapshot().read_known(128, 1024).unwrap(),
                    1_f32.to_le_bytes().repeat(256)
                );
                assert_eq!(l1.tracked_bytes(), 0);
                live.write_known_linear(128, &2_f32.to_le_bytes().repeat(256))
                    .unwrap();
            } else {
                assert_eq!(event.snapshot_address, None);
            }
            tick += 1;
        }
        assert_eq!(
            l1.read_known(0, 256).unwrap(),
            0x3c00_u16.to_le_bytes().repeat(128)
        );
    }
}
