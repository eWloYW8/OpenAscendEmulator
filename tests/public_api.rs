use open_ascend_emulator::memory::ub::UbMemory;
use open_ascend_emulator::sim::common::scalar::{ScalarMachine, ScalarMachineError, ScalarStepper};
use open_ascend_emulator::{Architecture, C220State};

#[test]
fn caller_can_construct_and_initialize_c220_state() -> Result<(), ScalarMachineError> {
    let mut machine = ScalarMachine::from_pem_initial_state(Architecture::Dav2201);
    machine.set_xreg(0, 0x2000)?;
    let mut state = C220State::new(
        ScalarStepper::new(machine, 0x1000),
        UbMemory::new(4096, 256),
    );
    state.scalar_mut().machine_mut().set_xreg(1, 7)?;
    assert_eq!(state.scalar().pc(), 0x1000);
    assert_eq!(&state.scalar().machine().xregs()[..2], &[0x2000, 7]);
    Ok(())
}
