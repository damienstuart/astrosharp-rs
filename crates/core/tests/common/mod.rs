//! Helpers shared by the parity tests against reference outputs from the original R code
//! (`tools/make_golden.R`).

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use astrosharp_core::{Image, Plane};

pub fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

/// The golden data directory, or `None` (with a note) when it has not been generated.
pub fn golden_dir() -> Option<PathBuf> {
    let dir = testdata().join("golden");
    if dir.join("cases.tsv").exists() {
        Some(dir)
    } else {
        eprintln!(
            "SKIPPED: no golden data; run `Rscript tools/make_golden.R <AstroSharp/resources/app> testdata/golden`"
        );
        None
    }
}

/// Reads an "ASAR" array written by make_golden.R: (dims in R order, row-major data).
pub fn read_arr(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..4], b"ASAR");
    let u32_at = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let ndim = u32_at(4);
    let dims: Vec<usize> = (0..ndim).map(|k| u32_at(8 + 4 * k)).collect();
    let data: Vec<f64> = b[8 + 4 * ndim..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    assert_eq!(data.len(), dims.iter().product::<usize>());
    (dims, data)
}

pub fn read_image_arr(path: &Path) -> Image {
    let (dims, data) = read_arr(path);
    let channels = if dims.len() == 3 { dims[2] } else { 1 };
    Image::new(dims[1], dims[0], channels, data)
}

/// An R matrix as a plane (R rows are plane rows).
pub fn read_plane(path: &Path) -> Plane {
    let (dims, data) = read_arr(path);
    assert_eq!(dims.len(), 2, "{}: not a matrix", path.display());
    Plane::new(dims[1], dims[0], data)
}

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Number of values that differ in their bits.
pub fn bit_mismatches(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}
