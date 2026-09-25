use super::{C220Load3dCoordinateError, C220Load3dV2Command};
use crate::isa::c220::mte::load3d::{C220Load3dDestination, C220Load3dElement};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220Load3dReadRequest {
    pub source_address: u32,
    pub input_bytes: u32,
    pub destination: C220Load3dDestination,
    pub destination_address: u32,
    pub output_bytes: u32,
    pub completes_output: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum C220Load3dRequestError {
    #[error(transparent)]
    Coordinates(#[from] C220Load3dCoordinateError),
    #[error("LOAD3D invalid K region has no preceding output request")]
    MissingPrecedingRequest,
}

impl C220Load3dV2Command {
    /// Logical UE3D requests, before physical L1 splitting and arbitration.
    pub fn read_requests(self) -> Result<Vec<C220Load3dReadRequest>, C220Load3dRequestError> {
        let mut requests: Vec<C220Load3dReadRequest> = Vec::new();
        let geometry = self.effective_geometry();
        let stride = u32::from(geometry.stride_w);
        let bytes = match self.operands.instruction.element {
            C220Load3dElement::B4 | C220Load3dElement::B8 => 1,
            C220Load3dElement::B16 => 2,
            C220Load3dElement::B32 => 4,
        };
        let block = 32 / bytes;
        for tile in self.tiles() {
            let mut coordinates = self.coordinates(tile)?.peekable();
            let mut destination = tile.destination_base as u32;
            let mut output_fill = 0u32;
            let mut invalid = coordinates.peek().is_some_and(|c| c.invalid_k);
            while let Some(first) = coordinates.next() {
                let mut row = [first; 16];
                for point in row.iter_mut().skip(1) {
                    *point = coordinates
                        .next()
                        .expect("coordinate rows contain sixteen points");
                }
                let width = first.channel_width;
                let mut merge_limit = if matches!(stride, 1 | 2 | 4 | 8) && width != 0 {
                    ((512 / width - 1) / stride + 1).min(16)
                } else {
                    1
                };
                let k_group = if block > width && geometry.dilation_w == 1 {
                    block
                        .div_ceil(width)
                        .min(first.filter_remaining)
                        .min(first.remaining_bytes)
                        .max(1)
                } else {
                    1
                };
                if block > width {
                    merge_limit = if stride == 8 || merge_limit == 1 {
                        1
                    } else {
                        (512 / (width * bytes * k_group)).min(16)
                    };
                }
                let mut source = if first.spatial_padding {
                    (first.source_base as u32).wrapping_add(
                        u32::from(self.matrix.width)
                            .wrapping_mul(u32::from(self.matrix.height))
                            .wrapping_mul(width)
                            .wrapping_mul(bytes)
                            .wrapping_mul(first.channel_block),
                    )
                } else {
                    first.source_address as u32
                };
                let next_row_first = coordinates.peek().copied();
                let mut skipped_all = false;
                for _ in 1..k_group {
                    for _ in 0..16 {
                        if coordinates.next().is_none() {
                            skipped_all = true;
                            break;
                        }
                    }
                    if skipped_all {
                        break;
                    }
                }
                let final_group = coordinates.peek().is_none();
                let mut merged = 0;
                for (point, coordinate) in row.iter().enumerate() {
                    let mut completes_output = false;
                    if point == 15 {
                        output_fill = output_fill.wrapping_add(width * bytes * k_group);
                        if output_fill >= 32 {
                            output_fill -= 32;
                            completes_output = true;
                        } else if output_fill != 0 && (first.last_k || final_group) {
                            output_fill = 0;
                            completes_output = true;
                        }
                    }
                    if invalid {
                        invalid = coordinate.invalid_k;
                    }
                    if !coordinate.row_wrap
                        && !coordinate.address_wrap
                        && !completes_output
                        && merge_limit > merged + 1
                    {
                        merged += 1;
                        continue;
                    }
                    if invalid {
                        if completes_output {
                            requests
                                .last_mut()
                                .ok_or(C220Load3dRequestError::MissingPrecedingRequest)?
                                .completes_output = true;
                        }
                    } else {
                        requests.push(C220Load3dReadRequest {
                            source_address: source,
                            input_bytes: (k_group + merged * stride) * width * bytes,
                            destination: self.operands.instruction.destination,
                            destination_address: destination,
                            output_bytes: 512,
                            completes_output,
                        });
                        invalid = row
                            .get(point + 1)
                            .or(next_row_first.as_ref())
                            .is_some_and(|c| c.invalid_k);
                    }
                    if let Some(next) = row.get(point + 1).or(next_row_first.as_ref()) {
                        source = next.source_address as u32;
                    }
                    merged = 0;
                    if completes_output {
                        destination = destination.wrapping_add(512);
                    }
                }
            }
        }
        Ok(requests)
    }
}
