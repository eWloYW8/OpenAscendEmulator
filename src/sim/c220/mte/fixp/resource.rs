use super::C220FixpCommand;
use crate::isa::c220::mte::fixp::C220FixpDestination;

/// Configuration retained until command retirement. Transfer addresses,
/// dimensions, factor values and NZ2ND selection do not change this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220FixpResourceKey {
    pub destination: C220FixpDestination,
    pub conversion: u8,
    pub activation: u8,
    pub saturation: bool,
}

impl C220FixpResourceKey {
    pub const fn new(command: C220FixpCommand, destination: C220FixpDestination) -> Self {
        Self {
            destination,
            conversion: command.descriptor.conversion_mode(),
            activation: command.descriptor.activation_mode(),
            saturation: command.control & (1 << 48) != 0,
        }
    }
}
