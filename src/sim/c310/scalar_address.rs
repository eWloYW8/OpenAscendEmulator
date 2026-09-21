const ADDRESS_ROOT_MASK: u64 = 0x1fffffe000000;
const BIU_ADDRESS_MASK: u64 = 0xffffffffffff;
const LOCAL_OFFSET_MASK: u64 = 0x7ffff;

pub(crate) const C310_UB_ROUTE_BYTES: u64 = 0x40000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum C310ScalarRoute {
    Hbm(u64),
    Ub(u64),
    Unsupported,
}

pub(crate) fn classify_c310_scalar_address(
    address: u64,
    spr67: u64,
    spr68: u64,
) -> C310ScalarRoute {
    let root = address & ADDRESS_ROOT_MASK;
    let spr67_root = spr67 & ADDRESS_ROOT_MASK;
    let spr68_root = spr68 & ADDRESS_ROOT_MASK;
    if root != spr67_root || address & 0x1000000 != 0 {
        return C310ScalarRoute::Hbm(address & BIU_ADDRESS_MASK);
    }
    let bank = (address >> 20) & 0x1f;
    if bank == 0 && address & 0x80000 != 0 {
        let offset = address & LOCAL_OFFSET_MASK;
        return if offset < C310_UB_ROUTE_BYTES {
            C310ScalarRoute::Ub(offset)
        } else {
            C310ScalarRoute::Unsupported
        };
    }
    if bank & 0xf == 0 {
        return C310ScalarRoute::Unsupported;
    }
    if spr67_root != spr68_root {
        return (address & 0xffffff)
            .checked_sub(0x100000)
            .and_then(|offset| (spr68 & BIU_ADDRESS_MASK).checked_add(offset))
            .map_or(C310ScalarRoute::Unsupported, C310ScalarRoute::Hbm);
    }
    let offset = address & LOCAL_OFFSET_MASK;
    if offset < C310_UB_ROUTE_BYTES {
        C310ScalarRoute::Ub(offset)
    } else {
        C310ScalarRoute::Unsupported
    }
}
