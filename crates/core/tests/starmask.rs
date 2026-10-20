//! Parity of the star-mask building blocks with R's imager (`tools/make_golden.R`).
//! Skipped when `testdata/golden` has not been generated.

mod common;

use astrosharp_core::starmask::{self, quantile7, sample_indices};
use astrosharp_core::{Plane, Progress, cimg, pbeta::pbeta, read_tiff};
use common::{bit_mismatches, golden_dir, max_abs_diff, read_arr, read_plane, testdata};

fn t1() -> Plane {
    let img = read_tiff(testdata().join("samples/T1.tif")).unwrap();
    Plane::new(img.width, img.height, img.data)
}

#[test]
fn blur_matches_imager() {
    let Some(dir) = golden_dir() else { return };
    let crop = read_plane(&dir.join("sm_crop.bin"));
    let (_, sigmas) = read_arr(&dir.join("sm_blur_sigmas.bin"));
    for s in sigmas {
        let mut ours = crop.clone();
        cimg::isoblur(&mut ours, s);
        // R formats the file names with as.character().
        let name = |kind: &str| dir.join(format!("sm_blur{kind}_{}.bin", r_num(s)));
        let nofma = read_plane(&name("_nofma"));
        assert_eq!(
            bit_mismatches(&ours.data, &nofma.data),
            0,
            "sigma {s}: differs from the plain-arithmetic transcription"
        );
        let imager = read_plane(&name(""));
        let d = max_abs_diff(&ours.data, &imager.data);
        println!("sigma {s:<6} max |diff| vs imager = {d:.3e}");
        assert!(d < 1e-11, "sigma {s}: {d:e}");
    }
}

/// R's as.character() for the sigmas used in the file names.
fn r_num(v: f64) -> String {
    let s = format!("{v}");
    s.strip_suffix(".0").map(str::to_string).unwrap_or(s)
}

#[test]
fn hessian_matches_imager_exactly() {
    let Some(dir) = golden_dir() else { return };
    let g = read_plane(&dir.join("sm_blur_4.bin"));
    let (xx, xy, yy) = cimg::hessian(&g);
    for (ours, name) in [(xx, "xx"), (xy, "xy"), (yy, "yy")] {
        let r = read_plane(&dir.join(format!("sm_hess_{name}.bin")));
        assert_eq!(bit_mismatches(&ours.data, &r.data), 0, "{name}");
    }
}

#[test]
fn response_matches() {
    let Some(dir) = golden_dir() else { return };
    let crop = read_plane(&dir.join("sm_crop.bin"));
    let ours = starmask::response(&crop, &Progress::new()).unwrap();
    let nofma = read_plane(&dir.join("sm_crop_D_nofma.bin"));
    // pmax and our maximum may disagree on the sign of an exact zero; compare values.
    let differing = ours
        .data
        .iter()
        .zip(&nofma.data)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(differing, 0, "response differs from the transcription");

    let ours = starmask::response(&t1(), &Progress::new()).unwrap();
    let imager = read_plane(&dir.join("sm_t1_D.bin"));
    let scale = imager.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let d = max_abs_diff(&ours.data, &imager.data) / scale;
    println!("T1 response: max |diff| / max |D| = {d:.3e}");
    assert!(d < 1e-9, "{d:e}");
}

#[test]
fn threshold_labels_and_mask_match() {
    let Some(dir) = golden_dir() else { return };
    let d_r = read_plane(&dir.join("sm_t1_D.bin"));
    let d_ours = starmask::response(&t1(), &Progress::new()).unwrap();
    let (dims, thr) = read_arr(&dir.join("sm_t1_thresholds.bin"));
    for row in thr.chunks_exact(dims[1]) {
        let (sensi, expected) = (row[0], row[1]);
        // Exact on R's own response; ours differs only by FMA-level noise.
        assert_eq!(starmask::threshold(&d_r, sensi), expected, "{sensi}%");
        let t = starmask::threshold(&d_ours, sensi);
        assert!(
            ((t - expected) / expected).abs() < 1e-8,
            "{sensi}%: {t} vs {expected}"
        );
        let flips = d_r
            .data
            .iter()
            .zip(&d_ours.data)
            .filter(|&(a, b)| (*a > expected) != (*b > t))
            .count();
        assert_eq!(flips, 0, "{sensi}%: thresholded pixels differ");
    }

    let thr98 = starmask::threshold(&d_r, 98.0);
    let b0 = Plane::new(
        d_r.width,
        d_r.height,
        d_r.data
            .iter()
            .map(|&v| f64::from(u8::from(v > thr98)))
            .collect(),
    );
    let labels = starmask::label(&b0);
    assert_eq!(labels, read_plane(&dir.join("sm_t1_label98.bin")));

    let mut bb = starmask::binary_mask(&d_ours, starmask::threshold(&d_ours, 98.0), 2.0);
    cimg::isoblur(&mut bb, 5.0);
    let bb_r = read_plane(&dir.join("sm_t1_Bb.bin"));
    let d = max_abs_diff(&bb.data, &bb_r.data);
    println!("feathered binary mask: max |diff| = {d:.3e}");
    assert!(d < 1e-11, "{d:e}");

    let m = starmask::mask(
        &d_ours,
        &starmask::MaskParams::default(),
        0..d_ours.height,
        0..d_ours.width,
    );
    let d = max_abs_diff(&m.data, &read_plane(&dir.join("sm_t1_M.bin")).data);
    println!("mask: max |diff| = {d:.3e}");
    assert!(d < 1e-9, "{d:e}");
}

#[test]
fn pbeta_matches_r() {
    let Some(dir) = golden_dir() else { return };
    let (dims, grid) = read_arr(&dir.join("pbeta_grid.bin"));
    let mut worst = (0.0, [0.0; 4]);
    for row in grid.chunks_exact(dims[1]) {
        let [x, a, b, p] = row.try_into().unwrap();
        let d = (pbeta(x, a, b) - p).abs();
        if d > worst.0 {
            worst = (d, [x, a, b, p]);
        }
    }
    println!("pbeta: max |diff| = {:.3e} at {:?}", worst.0, worst.1);
    assert!(worst.0 < 1e-12, "{worst:?}");
}

#[test]
fn sampling_and_quantile_match_r() {
    let Some(dir) = golden_dir() else { return };
    let (dims, rows) = read_arr(&dir.join("sample_idx.bin"));
    for row in rows.chunks_exact(dims[1]) {
        let n = row[0] as usize;
        let ours: Vec<f64> = sample_indices(n).iter().map(|&i| (i + 1) as f64).collect();
        assert_eq!(ours, row[1..], "n = {n}");
    }
    // Type-7 quantiles of a small vector, as quantile(1:10 / 7, p).
    let mut x: Vec<f64> = (1..=10).map(|v| v as f64 / 7.0).collect();
    let q = quantile7(&mut x, 0.98);
    let expected = (1.0 - 0.82) * (9.0 / 7.0) + 0.82 * (10.0 / 7.0);
    assert!((q - expected).abs() < 1e-15);
}
