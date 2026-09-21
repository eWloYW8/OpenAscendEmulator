#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220UbBank {
    pub group: u8,
    pub id: u8,
}

impl C220UbBank {
    pub const fn from_address(address: u64) -> Self {
        let group = ((address >> 5) & 0xf) as u8;
        let id = group | ((address >> 12) & 0x30) as u8;
        Self { group, id }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_id_uses_group_and_upper_region_bits() {
        assert_eq!(C220UbBank::from_address(0), C220UbBank { group: 0, id: 0 });
        assert_eq!(
            C220UbBank::from_address(0x20),
            C220UbBank { group: 1, id: 1 }
        );
        assert_eq!(
            C220UbBank::from_address(0x10020),
            C220UbBank { group: 1, id: 17 }
        );
    }
}
