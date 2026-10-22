//! `--progress` reports processing progress on stdout.

use std::path::{Path, PathBuf};
use std::process::Command;

fn sample() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/samples/T1.tif")
}

fn run(name: &str, extra: &[&str]) -> String {
    let out = std::env::temp_dir().join(format!(
        "astrosharp-progress-{}-{name}.tif",
        std::process::id()
    ));
    let result = Command::new(env!("CARGO_BIN_EXE_astrosharp"))
        .arg(sample())
        .arg(&out)
        .args(["--model", "dual-psf", "--star-psf", "2"])
        .args(extra)
        .output()
        .unwrap();
    let _ = std::fs::remove_file(&out);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    String::from_utf8(result.stdout).unwrap()
}

#[test]
fn progress_lines_rise_to_one() {
    let stdout = run("on", &["--progress"]);
    let values: Vec<f64> = stdout
        .lines()
        .map(|l| {
            let v = l
                .strip_prefix("progress ")
                .unwrap_or_else(|| panic!("unexpected line {l:?}"));
            v.parse().unwrap()
        })
        .collect();
    assert!(!values.is_empty());
    assert!(values.windows(2).all(|p| p[0] <= p[1]), "{values:?}");
    assert!(values.iter().all(|v| (0.0..=1.0).contains(v)), "{values:?}");
    assert_eq!(values.last(), Some(&1.0));
}

#[test]
fn stdout_is_empty_without_progress() {
    assert_eq!(run("off", &[]), "");
}
