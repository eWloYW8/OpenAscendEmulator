# OpenAscendEmulator

OpenAscendEmulator is a work-in-progress Rust implementation of selected
operator-simulator interfaces for `dav_2201` and `dav_3510`.

It currently supports configuration handling, isolated run preparation, and
selected instruction and memory operations. It is not a complete native
operator simulator; kernel execution, cycle accuracy, and report equivalence
are not yet available.

```bash
cargo test --all-targets
cargo run -- devices
cargo run -- op simulator --help
```

Use `cargo run -- --help` for the available commands and options. Running an
operator requires an appropriate local Ascend software installation and an
explicit `--execute` flag.
