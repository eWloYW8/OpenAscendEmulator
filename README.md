# OpenAscendEmulator

A work-in-progress Rust instruction and timing emulator for `dav_2201` and
`dav_3510`, currently focused on the C220/910B core.

## Organization

- `isa/`: instruction decoding and encoded operands, separated by architecture.
  C220 Vector families live in `isa/c220/vector/`; DMA, LOAD2D and BT
  encodings live in `isa/c220/mte/`.
- `sim/c220/core/`: instruction dispatch, execution-unit coordination, and program execution.
- `sim/c220/state/`: register/UB state, Vector-to-MTE3 handoff, and execution errors.
- `sim/c220/{cube,vector,mte,scalar}/`: unit-specific execution and timing rules.
- `sim/c220/{schedule,sync}.rs`: instruction issue clock, stalls, and hardware flags.
- `sim/c220/memory/`: C220 local-memory layout and resource arbitration.
  `l1/` owns bank arbitration, service responses and bounded transport;
  these mechanisms are separate private implementations behind one L1 API.
- `sim/c310/`: independent C310 execution components.
  `vector/` separates register state, arithmetic, predicates and memory transfers;
  implementation modules remain private behind the vector API.
  PUSH_PB and VF queue decoding live in `isa/c310/dispatch.rs`, not in runtime queues.
- `sim/common/`: mechanisms shared by implemented architectures.
  `scalar` exposes register state, stepping, events and the memory-bus
  contract; arithmetic helpers and bus adapters remain private.
- `image/`, `memory/`, `numeric/`: image loading, general memory storage, and numeric operations.

The C220 core owns the issue clock and coordinates shared memory and
cross-unit synchronization. Unit runtimes live with their execution and timing
rules, not inside the core. Vector instruction preparation, dispatch, and uop
planning belong to `vector/`; MTE transfer and flag transitions belong to `mte/`.
Execution units do not import the core, and ISA decoders do not import the simulator.
MTE2 owns its pending transfers and flags inside its pipeline; inspect these via
`core.mte2_pipeline()`. MTE state transitions use explicit register and memory
arguments instead of extending the whole-core state. Cross-unit barriers are
coordinated by the core. MTE3 reuses the transfer plan used for timing admission.

Cube separates MMAD traversal (`mmad.rs`), tiled memory addressing (`layout.rs`),
pure numerical operations (`numeric/`), and deferred output commits (`execute.rs`).
All supported MMAD formats share the traversal and output bookkeeping; their
slice widths, rounding, saturation, and nonfinite rules remain format-specific.
MTE1, MTE2, and MTE3 each own their transfers, execution state and timing under
`mte/{mte1,mte2,mte3}/`. Module entry points expose the unit API; timing,
transfer and state implementations stay private. MTE1 instruction families
have explicit `bias` and `load2d` APIs without duplicate unit-root exports.
`mte/interface/` owns shared L1 read arbitration, output scheduling and
L0 write interfaces. Only common DMA request expansion and transfer errors
remain at the MTE level. Cube numerical helpers are private; execution results
and timing state remain inspectable.

`mte/mte1/{bias,load2d}/` own data movement and lazy physical request expansion.
`mte/mte1/frontend.rs` provides independent BT and LOAD2D generation engines.
They share one `C220MteL1Interface` instead of owning separate read/output lanes.
The shared interface owns bounded input queues, unique request IDs, in-flight
response tracking and output scheduling. L1 bank arbitration and transport stay
external so other memory clients can participate.

The shared MTE L1 read arbiter exposes per-port eligibility and round-robin
selection, including output-fragment backpressure. LOAD2D has a lazy physical
read planner under `mte/mte1/load2d/`. Completing reads expand into L0 output
fragments using the same lazy output plan as BT. The L1 output scheduler owns
the shared acknowledgment FIFO, lazy expansion, destination backpressure and
BT retirement. A blocked target holds later responses behind it, including
responses for other destinations. Its generic payload retains the caller's
logical request without introducing an instruction trait. Both frontends feed
the same scheduler through the shared read interface.

Each L0A/L0B write interface owns its bounded input queues, priority, local
acknowledgments and instruction retirement. Callers forward emitted L0 fragments
to the indicated target and port; L1 output does not retire those instructions.
Cycle results expose sent fragments, blocked destinations, queue occupancy and
local retirements. Enqueue and step ordering is explicit so callback phases
are not hidden by the interface API.
Frontend idleness means generation is finished, not that submitted requests
have retired. The shared interface and destination write interfaces must also
drain. Cross-engine callback order remains explicit at the composition boundary;
complete callback-order equivalence is not established.
The integrated core still uses the aggregate timing lane; the L1 read and
L0 write components are not yet coordinated in core dispatch.

Vector has explicit boundaries between instruction preparation, operations,
operand reads, and scheduling:

```text
sim/c220/vector/
  instruction.rs    prepared instructions and their operand/store views
  dispatch.rs       instruction admission and register-state coordination
  issue.rs          register snapshots and operand preparation
  ops/              operation-specific planning and execution
  lanes/            data-type-specific lane evaluation (private)
  read/
    mod.rs          pending read state, UB ports, and captured bytes
    plan.rs         per-uop read planning (private)
    evaluate.rs     execution from captured operands (private)
  pipeline/
    mod.rs          cycle advancement, UB arbitration, and retirement
    admission.rs    validation and uop admission (private)
    hazards.rs      queue conflicts and inter-instruction gaps (private)
    updates.rs      ordered mask, reduction, and VA updates (private)
  timing.rs         operation stage latencies and writeback plans
  uop.rs            instruction-to-uop expansion
```

Control-word layouts live in `isa/c220/vector/`; decoding does not depend on
execution errors. Both architectures keep lane evaluators in the simulator.
Vector stores share one UB commit implementation; pipeline scheduling determines
when writes become visible. Queue classification uses the same rules at dispatch
and admission. Adding an instruction requires an explicit operand-read decision.
Old operation module paths are not retained as compatibility aliases.

Use `C220Core::new` with explicit timing rules, or `C220Core::with_config`
with `C220CoreConfig` to select the device and Cube configuration.
Initialize registers and UB through `C220State`; inspect them through `core.state()`.
Construct its scalar state using `sim::common::scalar::{ScalarMachine, ScalarStepper}`.
Vector events carry `C220VectorInstruction`, whose `uops()` and `write_plan()`
expose the unit's scheduled work.
The core exposes pipeline state, memory, execution events, and completion
results. Internal execution units are not part of the public API.

## Status

Instruction coverage and timing integration remain incomplete. Some timing
parameters must be supplied by the caller; complete cycle accuracy and
end-to-end equivalence are not established. C310 does not yet have a complete
integrated core.

## Development

```sh
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```
