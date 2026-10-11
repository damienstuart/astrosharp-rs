//! Parity tests against reference outputs produced by the original R code
//! (`tools/make_golden.R`). Skipped when `testdata/golden` has not been generated.

use std::path::{Path, PathBuf};

use astrosharp_core::features::OFFSETS_9X9;
use astrosharp_core::pipeline::r_preview_region;
use astrosharp_core::{
    Image, Mode, ModelKind, Params, Progress, color, process, read_tiff, rround,
};

/// Parity tolerance for the processed f64 values.
const TOL: f64 = 1e-9;

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn golden_dir() -> Option<PathBuf> {
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
fn read_arr(path: &Path) -> (Vec<usize>, Vec<f64>) {
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

fn read_image_arr(path: &Path) -> Image {
    let (dims, data) = read_arr(path);
    let channels = if dims.len() == 3 { dims[2] } else { 1 };
    Image::new(dims[1], dims[0], channels, data)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn feature_order_matches_getmatrix9() {
    let Some(dir) = golden_dir() else { return };
    let text = std::fs::read_to_string(dir.join("offsets_9x9.txt")).unwrap();
    let r: Vec<(i8, i8)> = text
        .lines()
        .map(|l| {
            let mut it = l.split_whitespace().map(|v| v.parse().unwrap());
            (it.next().unwrap(), it.next().unwrap())
        })
        .collect();
    assert_eq!(r, OFFSETS_9X9);
}

#[test]
fn round_matches_r() {
    let Some(dir) = golden_dir() else { return };
    let (dims, data) = read_arr(&dir.join("round5.bin"));
    let mismatches: Vec<_> = data
        .chunks_exact(dims[1])
        .filter(|p| rround::round_digits(p[0], 5) != p[1])
        .map(|p| (p[0], p[1], rround::round_digits(p[0], 5)))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{} mismatches, e.g. {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)]
    );
}

#[test]
fn color_conversion_matches_r() {
    let Some(dir) = golden_dir() else { return };
    let rgb = read_arr(&dir.join("color_rgb.bin")).1;
    let luv = read_arr(&dir.join("color_luv.bin")).1;
    let luv_mod = read_arr(&dir.join("color_luv_mod.bin")).1;
    let rgb_back = read_arr(&dir.join("color_rgb_back.bin")).1;

    let ours: Vec<f64> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&p| color::srgb_to_luv(p))
        .collect();
    let d = max_abs_diff(&ours, &luv);
    assert!(d < 1e-10, "sRGB -> Luv max diff {d}");

    let ours: Vec<f64> = luv_mod
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&p| color::luv_to_srgb(p))
        .collect();
    let d = max_abs_diff(&ours, &rgb_back);
    assert!(d < 1e-12, "Luv -> sRGB max diff {d}");
}

#[test]
fn tiff_reading_matches_readtiff() {
    let Some(dir) = golden_dir() else { return };
    for name in ["T1.tif", "CT1.tif", "R1.tif", "CR1.tif"] {
        let ours = read_tiff(testdata().join("samples").join(name)).unwrap();
        let r = read_image_arr(&dir.join(format!("input_{name}.bin")));
        assert_eq!(
            (ours.width, ours.height, ours.channels),
            (r.width, r.height, r.channels),
            "{name}"
        );
        assert_eq!(ours.data, r.data, "{name}");
    }
}

#[test]
fn pipeline_matches_r() {
    let Some(dir) = golden_dir() else { return };
    let manifest = std::fs::read_to_string(dir.join("cases.tsv")).unwrap();
    let mut failures = Vec::new();
    for line in manifest.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let [name, image, color, model, psf, aggr, seg, preview, px, py] = f[..] else {
            panic!("bad manifest line: {line}");
        };
        let model = match model {
            "PSF Model (Pre-Beta)" => ModelKind::Psf,
            "AstroClean" => ModelKind::AstroClean,
            "Hybrid Model" => ModelKind::Hybrid,
            other => panic!("unexpected model {other}"),
        };
        let mut img = read_tiff(testdata().join("samples").join(image)).unwrap();
        assert_eq!(img.is_color(), color == "Color");
        if preview == "Preview" {
            let (rows, cols) = r_preview_region(
                img.height,
                img.width,
                px.parse().unwrap(),
                py.parse().unwrap(),
            );
            img = img.crop(rows, cols);
        }
        let params = Params {
            model,
            psf_fwhm: psf.parse().unwrap(),
            aggressiveness: aggr.parse().unwrap(),
            mode: Mode::Parity,
            chunk_size: seg.parse().unwrap(),
        };
        let ours = process(&img, &params, &Progress::new()).unwrap();
        let expected = read_image_arr(&dir.join(format!("{name}.bin")));
        assert_eq!(
            (ours.width, ours.height, ours.channels),
            (expected.width, expected.height, expected.channels),
            "{name}: shape"
        );
        let d = max_abs_diff(&ours.data, &expected.data);
        println!("{name:<22} max |diff| = {d:.3e}");
        if d > TOL {
            failures.push(format!("{name}: {d:e}"));
        }
    }
    assert!(failures.is_empty(), "parity failures: {failures:?}");
}

#[test]
fn seamless_matches_parity_for_single_models() {
    let img = read_tiff(testdata().join("samples/T1.tif")).unwrap();
    for model in [ModelKind::Psf, ModelKind::AstroClean] {
        let p = Params {
            model,
            chunk_size: 150,
            ..Params::default()
        };
        let a = process(
            &img,
            &Params {
                mode: Mode::Parity,
                ..p.clone()
            },
            &Progress::new(),
        )
        .unwrap();
        let b = process(
            &img,
            &Params {
                mode: Mode::Seamless,
                ..p
            },
            &Progress::new(),
        )
        .unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn seamless_colour_keeps_full_precision() {
    let img = read_tiff(testdata().join("samples/CT1.tif")).unwrap();
    // Aggressiveness 0 leaves lightness unchanged, so only the colour round trip remains.
    let p = Params {
        model: ModelKind::Psf,
        aggressiveness: 0.0,
        ..Params::default()
    };
    let run = |mode| process(&img, &Params { mode, ..p.clone() }, &Progress::new()).unwrap();
    let seamless = max_abs_diff(&run(Mode::Seamless).data, &img.data);
    assert!(
        seamless < 1e-9,
        "seamless colour round trip changed values by {seamless}"
    );
    // Parity mode rounds to 5 decimals like R's convertColor.
    let parity = max_abs_diff(&run(Mode::Parity).data, &img.data);
    assert!(
        parity > 1e-7 && parity <= 5e-6 + 1e-12,
        "parity diff {parity}"
    );
}

#[test]
fn sample_images_are_not_flagged_as_linear() {
    for name in ["T1.tif", "CT1.tif", "R1.tif", "CR1.tif"] {
        let img = read_tiff(testdata().join("samples").join(name)).unwrap();
        assert!(!img.looks_linear(), "{name}");
    }
}

#[test]
fn hybrid_seams_are_confined_to_chunk_boundaries() {
    let img = read_tiff(testdata().join("samples/T1.tif")).unwrap();
    let p = Params {
        model: ModelKind::Hybrid,
        chunk_size: 150,
        ..Params::default()
    };
    let parity = process(
        &img,
        &Params {
            mode: Mode::Parity,
            ..p.clone()
        },
        &Progress::new(),
    )
    .unwrap();
    let seamless = process(
        &img,
        &Params {
            mode: Mode::Seamless,
            ..p
        },
        &Progress::new(),
    )
    .unwrap();
    // Chunk breaks in padded coordinates are 1, 151, 301, 451; internal boundaries sit
    // around original index 150k - 4 (0-based). Differences must stay within 8 px of them.
    let near_boundary = |i: usize| (1..4).any(|k| (i as i64 - (150 * k as i64 - 4)).abs() <= 8);
    let mut differing = 0;
    for r in 0..img.height {
        for c in 0..img.width {
            let i = r * img.width + c;
            if parity.data[i] != seamless.data[i] {
                differing += 1;
                assert!(
                    near_boundary(r) || near_boundary(c),
                    "unexpected difference at ({r}, {c})"
                );
            }
        }
    }
    assert!(differing > 0, "expected chunk seams in parity mode");
}
