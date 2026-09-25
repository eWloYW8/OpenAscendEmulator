mod compare;
mod execute;
mod load;
mod preload;
pub use preload::C220PreloadOperands;
mod store;
pub use compare::{C220ScalarFp32CompareOutcome, execute_fp32_compare_word};
pub use load::C220LoadOperands;
pub use store::{
    C220AtomicStoreError, C220AtomicStoreOperands, C220AtomicStoreResult, C220DirectStoreOperands,
    C220StoreOperands,
};
mod float;
pub use float::{C220ScalarFp32Outcome, execute_fp32_word};
pub mod spr;
use crate::architecture::Architecture;
use crate::isa::c220::vector::C220MovemaskHint;
use crate::sim::common::scalar::{ScalarMachine, ScalarMachineError};
mod conversion;
pub use conversion::{
    C220ScalarConversionOutcome, C220ScalarConversionStatus, execute_conversion_word,
    execute_scalar_conversion,
};

pub(crate) mod address;
pub use address::{C220ScalarAddressConfig, C220ScalarMappedAddress};
pub(crate) mod bus;
pub mod lsu;
pub mod timing;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct C220MovemaskStep {
    pub pc: u64,
    pub word: u32,
    pub source_register: u8,
    pub source_value: u64,
    pub destination_spr: u16,
    pub prior_destination_value: Option<u64>,
}

pub fn execute_movemask(
    machine: &mut ScalarMachine,
    pc: u64,
    word: u32,
) -> Result<C220MovemaskStep, ScalarMachineError> {
    if machine.architecture() != Architecture::Dav2201 {
        return Err(ScalarMachineError::UnsupportedWord { pc, word });
    }
    let hint = C220MovemaskHint::from_word(word)
        .ok_or(ScalarMachineError::UnsupportedWord { pc, word })?;
    let source_value = machine.xregs()[usize::from(hint.source_register)];
    let prior_destination_value = machine.spr_value(hint.destination_spr);
    machine.set_spr_value(hint.destination_spr, source_value)?;
    Ok(C220MovemaskStep {
        pc,
        word,
        source_register: hint.source_register,
        source_value,
        destination_spr: hint.destination_spr,
        prior_destination_value,
    })
}
