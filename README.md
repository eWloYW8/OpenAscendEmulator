# OpenAscendEmulator

OpenAscendEmulator is a work-in-progress Rust instruction and timing emulator
for `dav_2201` and `dav_3510`.

The current focus is the `dav_2201` core. Its loaded-program execution path
connects scalar, selected vector, and memory-transfer instructions with an
explicit MTE2 timeline. The 16- and 32-bit MOVEV paths decode register
operands and respect active lanes; vector stores also report their UB bank
placement. Single-repeat vector instructions use their encoded per-block UB
strides instead of requiring one fixed control word.
Vector execution exposes per-block UB write demand and an uncontended
writeback-latency fragment; bank conflicts and whole-pipeline timing remain
unmodeled.
The supported C220 MOV paths handle independent source and destination burst
gaps, including segmented output writes.
MTE timing rules must be supplied by the caller; cycle accuracy and full
instruction coverage are not yet available.

```bash
cargo check --all-targets
```
