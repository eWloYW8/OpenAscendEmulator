use std::process::ExitCode;

fn main() -> ExitCode {
    match open_ascend_emulator::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("open-ascend-emulator: {error}");
            ExitCode::FAILURE
        }
    }
}
