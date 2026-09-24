const ADDRESS_ROOT_MASK: u64 = 0x1fffffe000000;
const LOCAL_OFFSET_MASK: u64 = 0x7ffff;
const EXTERNAL_ADDRESS_MASK: u64 = 0xffffffffffff;

pub(crate) const C220_UB_BYTES: u64 = 0x30000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C220ScalarRoute {
    Hbm(u64),
    Ub(u64),
    Unsupported,
}

pub(crate) fn classify_c220_scalar_address(
    address: u64,
    spr67: u64,
    spr68: u64,
) -> C220ScalarRoute {
    let root = address & ADDRESS_ROOT_MASK;
    if address & 0x1000000 != 0 || root != spr67 & ADDRESS_ROOT_MASK {
        return C220ScalarRoute::Hbm(address & EXTERNAL_ADDRESS_MASK);
    }
    let bank = (address >> 20) & 0x1f;
    if bank == 0 && address & 0x80000 != 0 {
        return C220ScalarRoute::Ub(address & LOCAL_OFFSET_MASK);
    }
    if bank & 0xf == 0 {
        return C220ScalarRoute::Unsupported;
    }
    if (spr67 & ADDRESS_ROOT_MASK) != (spr68 & ADDRESS_ROOT_MASK) {
        return C220ScalarRoute::Hbm(
            (spr68 & EXTERNAL_ADDRESS_MASK)
                .wrapping_sub(0x100000)
                .wrapping_add(address & 0xffffff),
        );
    }
    let offset = address & LOCAL_OFFSET_MASK;
    if offset < C220_UB_BYTES {
        C220ScalarRoute::Ub(offset)
    } else {
        C220ScalarRoute::Unsupported
    }
}
