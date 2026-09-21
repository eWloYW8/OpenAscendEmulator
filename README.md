# OpenAscendEmulator

OpenAscendEmulator is a work-in-progress Rust instruction and timing emulator
for `dav_2201` and `dav_3510`.

The current focus is the `dav_2201` core. Its loaded-program execution path
connects scalar, selected vector, and memory-transfer instructions with an
explicit MTE2 timeline. The 16- and 32-bit MOVEV paths decode register
operands and respect active lanes; vector stores also report their UB bank
placement. MOVEV and FP32 add/subtract/multiply/divide/maximum/minimum use
independently encoded per-block and per-repeat UB strides; count masks can
cover multiple repeats with a final partial tile. FP32 VABS and VRELU use the
unary stride layout and one UB read source. Their modeled vector execution
stages take 15 and 6 ticks, respectively.
Vector execution exposes per-block UB write demand, bank-group writeback
latency, and per-uop read/execute delays. The full-block writeback estimate
splits writes at 32-byte UB boundaries and arbitrates bank and bank-group
occupancy.
The timed core defers 16- and 32-bit MOVEV and FP32 results until their modeled
UB response. MTE3 completion waits
drain the vector queue; dependent vector reads require explicit synchronization.
FP32 operands are sampled from UB when their read blocks win arbitration,
after instruction issue. Fully masked source blocks are not read and appear as
zeroes in the operand snapshot. Competing read ports and vector writes share
the same per-cycle UB bank arbiter and can delay subsequent execution and write
visibility. Partial-write occupancy and other UB clients are not yet fully
integrated into this timeline.
Dispatch and UB response delays are caller-supplied parameters.
The supported C220 MOV paths handle independent source and destination burst
gaps, including segmented output writes.
Timing rules must be supplied by the caller; cycle accuracy and full
instruction coverage are not yet available.

```bash
cargo check --all-targets
```
