use crate::sim::c220::memory::{C220LocalBuffer, C220LocalBufferError};

pub(super) const TILE_EDGE: u64 = 16;
pub(super) const F16_TILE_BYTES: u64 = 512;
pub(super) const F32_INPUT_K_TILE: u64 = 8;
pub(super) const F32_INPUT_TILE_BYTES: u64 = 512;
pub(super) const F32_TILE_BYTES: u64 = 1024;

pub(super) fn f16_b_address(base: u64, n_tiles: u64, k: u64, n: u64) -> u64 {
    let tile = (k / TILE_EDGE) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * TILE_EDGE + k % TILE_EDGE;
    base.wrapping_add(tile * F16_TILE_BYTES + lane * 2)
}

pub(super) fn f32_b_address(base: u64, n_tiles: u64, k: u64, n: u64) -> u64 {
    let tile = (k / F32_INPUT_K_TILE) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * F32_INPUT_K_TILE + k % F32_INPUT_K_TILE;
    base.wrapping_add(tile * F32_INPUT_TILE_BYTES + lane * 4)
}

pub(super) fn integer_a_element(k_tiles: u64, k_tile: u64, m: u64, k: u64) -> u64 {
    let tile = (m / TILE_EDGE) * k_tiles + k / k_tile;
    let lane = (m % TILE_EDGE) * k_tile + k % k_tile;
    tile * TILE_EDGE * k_tile + lane
}

pub(super) fn integer_b_element(n_tiles: u64, k_tile: u64, k: u64, n: u64) -> u64 {
    let tile = (k / k_tile) * n_tiles + n / TILE_EDGE;
    let lane = (n % TILE_EDGE) * k_tile + k % k_tile;
    tile * TILE_EDGE * k_tile + lane
}

pub(super) fn f32_c_address(base: u64, m_tiles: u64, m: u64, n: u64) -> u64 {
    let tile = m / TILE_EDGE + m_tiles * (n / TILE_EDGE);
    let lane = (m % TILE_EDGE) * TILE_EDGE + n % TILE_EDGE;
    output_tile_base(base, tile, F32_TILE_BYTES) + lane * 4
}

pub(super) fn f16_c_address(base: u64, m_tiles: u64, m: u64, n: u64) -> u64 {
    let tile = m / TILE_EDGE + m_tiles * (n / TILE_EDGE);
    let lane = (m % TILE_EDGE) * TILE_EDGE + n % TILE_EDGE;
    output_tile_base(base, tile, F16_TILE_BYTES) + lane * 2
}

fn output_tile_base(base: u64, tile: u64, bytes: u64) -> u64 {
    let address = base.wrapping_add(tile * bytes);
    if address > 131072 - bytes {
        address % 131072
    } else {
        address
    }
}

#[cfg(test)]
pub(super) fn read_u16_wrapped(
    buffer: &C220LocalBuffer,
    address: u64,
) -> Result<u16, C220LocalBufferError> {
    let bytes = buffer.read_known_wrapped(address, 2)?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

pub(super) fn read_input<const N: usize>(
    buffer: &C220LocalBuffer,
    base: u64,
    offset: u64,
) -> Result<[u8; N], C220LocalBufferError> {
    let tile = base.wrapping_add(offset / 512 * 512);
    let tile = if tile > 65536 - 512 {
        tile % 65536
    } else {
        tile
    };
    let bytes = buffer.read_initialized_linear(tile + offset % 512, N)?;
    Ok(std::array::from_fn(|index| bytes[index]))
}

#[cfg(test)]
pub(super) fn read_u32_wrapped(
    buffer: &C220LocalBuffer,
    address: u64,
) -> Result<u32, C220LocalBufferError> {
    let bytes = buffer.read_known_wrapped(address, 4)?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

#[cfg(test)]
pub(super) fn write_u32_wrapped(
    buffer: &mut C220LocalBuffer,
    address: u64,
    value: u32,
) -> Result<(), C220LocalBufferError> {
    buffer.write_known_wrapped(address, &value.to_le_bytes())
}

#[cfg(test)]
pub(super) fn write_u16_wrapped(
    buffer: &mut C220LocalBuffer,
    address: u64,
    value: u16,
) -> Result<(), C220LocalBufferError> {
    buffer.write_known_wrapped(address, &value.to_le_bytes())
}
