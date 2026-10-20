//! Processing a region gives exactly that part of the full result, and the properties of
//! the star mask models that don't need R.

use std::ops::Range;
use std::path::Path;

use astrosharp_core::{
    Image, Keep, MaskParams, Mode, ModelKind, Params, Progress, Reuse, process, process_region,
    process_with, read_tiff,
};

fn sample(name: &str) -> Image {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/samples")
        .join(name);
    read_tiff(path).unwrap().crop(90..250, 110..300)
}

fn bits(img: &Image) -> Vec<u64> {
    img.data.iter().map(|v| v.to_bits()).collect()
}

fn params(model: ModelKind, mode: Mode) -> Params {
    Params {
        model,
        mode,
        star_psf_fwhm: 2.0,
        chunk_size: 60,
        ..Params::default()
    }
}

/// Regions in the middle, at the corners and along the edges (including some smaller
/// than the margin).
fn regions(img: &Image) -> Vec<(Range<usize>, Range<usize>)> {
    let (w, h) = (img.width, img.height);
    vec![
        (40..90, 50..120),
        (0..30, 0..25),
        (h - 3..h, w - 2..w),
        (0..h, w - 40..w),
        (70..71, 3..4),
    ]
}

fn region_of(
    img: &Image,
    p: &Params,
    rows: Range<usize>,
    cols: Range<usize>,
    reuse: Reuse,
) -> Image {
    process_region(img, rows, cols, p, reuse, Keep::NONE, &Progress::new())
        .unwrap()
        .0
}

#[test]
fn regions_match_the_full_result() {
    let mut cases = Vec::new();
    for model in ModelKind::ALL {
        cases.push(params(model, Mode::Seamless));
    }
    for model in [ModelKind::DualPsf, ModelKind::StarMask] {
        cases.push(params(model, Mode::Parity));
    }
    for image in ["T1.tif", "CT1.tif"] {
        let img = sample(image);
        for p in &cases {
            let full = process(&img, p, &Progress::new()).unwrap();
            for (rows, cols) in regions(&img) {
                let ours = region_of(&img, p, rows.clone(), cols.clone(), Reuse::default());
                let expected = full.crop(rows.clone(), cols.clone());
                assert_eq!(
                    (ours.width, ours.height, ours.channels),
                    (expected.width, expected.height, expected.channels)
                );
                assert_eq!(
                    bits(&ours),
                    bits(&expected),
                    "{image} {} {rows:?} x {cols:?}",
                    p.describe()
                );
            }
        }
    }
}

#[test]
fn region_outputs_are_reused_only_for_the_same_region() {
    let img = sample("T1.tif");
    let p = params(ModelKind::DualPsf, Mode::Seamless);
    let (rows, cols) = (40..90, 50..120);
    let (out, fresh) = process_region(
        &img,
        rows.clone(),
        cols.clone(),
        &p,
        Reuse::default(),
        Keep::ALL,
        &Progress::new(),
    )
    .unwrap();
    let nets = fresh.nets.unwrap();
    let response = fresh.response.unwrap();
    assert!(nets.matches_region(&img, rows.clone(), cols.clone(), &p));
    assert!(!nets.matches_region(&img, 41..91, cols.clone(), &p));
    assert!(!nets.matches(&img, &p));

    // New mask settings and aggressiveness: only the mask and blend are redone.
    let q = Params {
        aggressiveness: 0.9,
        mask: MaskParams {
            sensitivity: 95.0,
            feathering: 3.0,
            ..MaskParams::default()
        },
        ..p.clone()
    };
    let reuse = Reuse {
        nets: Some(&nets),
        response: Some(&response),
    };
    let (reused, fresh) = process_region(
        &img,
        rows.clone(),
        cols.clone(),
        &q,
        reuse,
        Keep::ALL,
        &Progress::new(),
    )
    .unwrap();
    assert!(fresh.nets.is_none() && fresh.response.is_none());
    assert_eq!(
        bits(&reused),
        bits(&region_of(
            &img,
            &q,
            rows.clone(),
            cols.clone(),
            Reuse::default()
        ))
    );
    assert_ne!(bits(&reused), bits(&out));

    // A different star PSF needs new network outputs, but not a new response.
    let s = Params {
        star_psf_fwhm: 2.5,
        ..p
    };
    let (_, fresh) =
        process_region(&img, rows, cols, &s, reuse, Keep::ALL, &Progress::new()).unwrap();
    assert!(fresh.nets.is_some() && fresh.response.is_none());
}

#[test]
fn response_is_reused_across_models_modes_and_regions() {
    let img = sample("CT1.tif");
    let (_, fresh) = process_with(
        &img,
        &params(ModelKind::StarMask, Mode::Seamless),
        Reuse::default(),
        Keep::ALL,
        &Progress::new(),
    )
    .unwrap();
    let response = fresh.response.expect("Star Mask computes the response");
    let reuse = Reuse {
        nets: None,
        response: Some(&response),
    };
    for p in [
        params(ModelKind::StarMask, Mode::Parity),
        params(ModelKind::DualPsf, Mode::Seamless),
    ] {
        let (out, fresh) = process_with(&img, &p, reuse, Keep::ALL, &Progress::new()).unwrap();
        assert!(fresh.response.is_none(), "{}", p.describe());
        assert_eq!(
            bits(&out),
            bits(&process(&img, &p, &Progress::new()).unwrap())
        );
        let ours = region_of(&img, &p, 10..60, 20..90, reuse);
        assert_eq!(bits(&ours), bits(&out.crop(10..60, 20..90)));
    }
    // A response of another image is not used.
    let other = img.crop(0..100, 0..100);
    let p = params(ModelKind::StarMask, Mode::Seamless);
    let (_, fresh) = process_with(&other, &p, reuse, Keep::ALL, &Progress::new()).unwrap();
    assert!(fresh.response.is_some());
}

#[test]
fn star_mask_of_grey_images_is_the_same_in_both_modes() {
    let img = sample("T1.tif");
    let run = |mode| process(&img, &params(ModelKind::StarMask, mode), &Progress::new()).unwrap();
    assert_eq!(bits(&run(Mode::Seamless)), bits(&run(Mode::Parity)));
    // Colour: one mask channel in seamless mode, grey RGB like R in parity mode.
    let img = sample("CT1.tif");
    let run = |mode| process(&img, &params(ModelKind::StarMask, mode), &Progress::new()).unwrap();
    assert_eq!(run(Mode::Seamless).channels, 1);
    let rgb = run(Mode::Parity);
    assert_eq!(rgb.channels, 3);
    let grey = |p: &[f64; 3]| p[0] == p[1] && p[1] == p[2];
    assert!(rgb.data.as_chunks::<3>().0.iter().all(grey));
}

#[test]
fn seamless_dual_psf_mixes_the_two_psf_results_through_the_mask() {
    let img = sample("T1.tif");
    let full = Params {
        aggressiveness: 1.0,
        ..params(ModelKind::DualPsf, Mode::Seamless)
    };
    let dual = process(&img, &full, &Progress::new()).unwrap();
    let psf = |fwhm| {
        let p = Params {
            model: ModelKind::Psf,
            psf_fwhm: fwhm,
            ..full.clone()
        };
        process(&img, &p, &Progress::new()).unwrap()
    };
    let (dso, star) = (psf(full.psf_fwhm), psf(full.star_psf_fwhm));
    let mask = process(
        &img,
        &Params {
            model: ModelKind::StarMask,
            ..full.clone()
        },
        &Progress::new(),
    )
    .unwrap();
    let (w, h) = (img.width, img.height);
    for r in 0..h {
        for c in 0..w {
            let i = r * w + c;
            let x = img.data[i];
            let kept = (4..h - 5).contains(&r) && (4..w - 5).contains(&c);
            let expected = if kept {
                let m = mask.data[i];
                star.data[i] * m + dso.data[i] * (1.0 - m)
            } else {
                x
            };
            assert_eq!(dual.data[i], expected.clamp(0.0, 1.0), "({r}, {c})");
        }
    }
    // Some pixels take the star model and some the DSO model.
    assert!(mask.data.iter().any(|&m| m > 0.99) && mask.data.iter().any(|&m| m < 0.01));
}

#[test]
fn seamless_dual_psf_does_not_depend_on_the_chunk_size() {
    let img = sample("T1.tif");
    let run = |chunk_size, mode| {
        let p = Params {
            chunk_size,
            ..params(ModelKind::DualPsf, mode)
        };
        process(&img, &p, &Progress::new()).unwrap()
    };
    assert_eq!(
        bits(&run(40, Mode::Seamless)),
        bits(&run(150, Mode::Seamless))
    );
    // In parity mode each chunk has its own mask.
    assert_ne!(bits(&run(40, Mode::Parity)), bits(&run(150, Mode::Parity)));
}

#[test]
fn invalid_mask_settings_are_rejected() {
    let img = sample("T1.tif");
    for mask in [
        MaskParams {
            sensitivity: 101.0,
            ..MaskParams::default()
        },
        MaskParams {
            feathering: -1.0,
            ..MaskParams::default()
        },
        MaskParams {
            blend: 0.0,
            ..MaskParams::default()
        },
    ] {
        let p = Params {
            mask,
            ..params(ModelKind::StarMask, Mode::Seamless)
        };
        assert!(process(&img, &p, &Progress::new()).is_err(), "{mask:?}");
    }
    let p = Params {
        star_psf_fwhm: 3.1,
        ..params(ModelKind::DualPsf, Mode::Seamless)
    };
    assert!(process(&img, &p, &Progress::new()).is_err());
}
