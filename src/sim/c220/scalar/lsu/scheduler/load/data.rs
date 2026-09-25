use super::*;
use crate::sim::c220::scalar::lsu::store_buffer::{C220LsuPairPart, C220LsuStoreEntry};

impl PendingLoad {
    pub(super) fn access_bytes(&self) -> usize {
        if self.part == C220LsuPairPart::Both {
            self.operands.access_bytes()
        } else {
            usize::from(self.operands.width_bytes)
        }
    }

    pub(super) fn read_line(&self, line: &[u8]) -> [u64; 2] {
        let width = usize::from(self.operands.width_bytes);
        let mut values = [0; 2];
        for (index, value) in values.iter_mut().enumerate() {
            let start = match (self.part, index) {
                (C220LsuPairPart::Both, 0) => self.offset,
                (C220LsuPairPart::Both, 1) if self.operands.second_destination.is_some() => {
                    self.offset + width
                }
                (C220LsuPairPart::First, 0) | (C220LsuPairPart::Second, 1) => self.offset,
                _ => continue,
            };
            let mut bytes = [0; 8];
            bytes[..width].copy_from_slice(&line[start..start + width]);
            *value = u64::from_le_bytes(bytes);
        }
        values
    }

    pub(super) fn read_data(
        &self,
        line: Option<&[u8]>,
        store: Option<&C220LsuStoreEntry>,
        covered: usize,
    ) -> Result<([u64; 2], bool), C220LsuSchedulerError> {
        let width = usize::from(self.operands.width_bytes);
        let mut values = line.map(|line| self.read_line(line)).unwrap_or([0; 2]);
        let Some(store) = store else {
            return Ok((values, false));
        };
        let mut forwarded = false;
        if self.operands.second_destination.is_some() {
            // A hit overlays only partial single-operand coverage or complete
            // pair coverage. Intermediate coverage leaves the cache data intact.
            if (covered > 0 && covered <= width) || covered == 2 * width {
                forwarded = store
                    .forward_pair(self.offset, width, self.part, &mut values)?
                    .into_iter()
                    .any(|changed| changed);
            }
        } else if covered != 0 {
            let mut bytes = values[0].to_le_bytes();
            store.forward_scalar(self.offset, &mut bytes[..width])?;
            values[0] = u64::from_le_bytes(bytes);
            forwarded = true;
        }
        Ok((values, forwarded))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::architecture::Architecture;
    use crate::sim::c220::scalar::lsu::store_buffer::{
        C220LsuMemory, C220LsuStoreBuffer, C220LsuStoreConfig,
    };
    use crate::sim::common::scalar::ScalarMachine;

    #[test]
    fn paired_hit_overlay_obeys_coverage_and_operand_start_bits() {
        for dtype in 0..4 {
            let width = 1_usize << dtype;
            for covered in [0, 1, width, width + 1, 2 * width] {
                let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
                machine.set_xreg(5, 0x108).unwrap();
                let word = (9 << 24) | (dtype << 22) | (7 << 17) | (5 << 12) | (8 << 7);
                let operands = C220LoadOperands::capture(&machine, 0, word).unwrap();
                let key = C220LsuLineKey {
                    address: 0x100,
                    memory: C220LsuMemory::External,
                };
                let pending = PendingLoad {
                    operands,
                    mapped: C220ScalarMappedAddress {
                        address: 0x108,
                        memory: key.memory,
                        stack: false,
                    },
                    line: key,
                    partition_address: 0x108,
                    offset: 8,
                    lookup: None,
                    part: C220LsuPairPart::Both,
                };
                let mut stores = C220LsuStoreBuffer::new(C220LsuStoreConfig {
                    line_bytes: 64,
                    main_entries: 1,
                    sub_entries: 2,
                    timeout_ticks: 4,
                })
                .unwrap();
                if covered != 0 {
                    stores
                        .store(key, C220LsuRequestId(0), 8, &vec![0xaa; covered], true)
                        .unwrap();
                }
                let (values, forwarded) = pending
                    .read_data(Some(&[0x55; 64]), stores.entry(key), covered)
                    .unwrap();
                let mut expected = [[0; 8]; 2];
                expected[0][..width].fill(0x55);
                expected[1][..width].fill(0x55);
                let overlay = covered != 0 && (covered <= width || covered == 2 * width);
                if overlay {
                    expected[0][..width].fill(0);
                    expected[0][..width.min(covered)].fill(0xaa);
                    if covered == 2 * width {
                        expected[1][..width].fill(0xaa);
                    }
                }
                assert_eq!(forwarded, overlay);
                assert_eq!(values, expected.map(u64::from_le_bytes));
            }
        }
    }
}
