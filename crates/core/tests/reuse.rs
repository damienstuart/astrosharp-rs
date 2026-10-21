//! Reusing network outputs with `process_with` gives exactly what `process` computes.

use std::path::Path;

use astrosharp_core::{
    Error, Image, Keep, Mode, ModelKind, NetOutputs, Params, Progress, Reuse, net_outputs_bytes,
    process, process_with, read_tiff,
};

fn sample(name: &str) -> Image {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../testdata/samples")
        .join(name);
    // A crop keeps the many runs below quick.
    read_tiff(path).unwrap().crop(100..260, 120..300)
}

/// `process_with`, reusing and keeping only network outputs.
fn with_nets(
    img: &Image,
    params: &Params,
    cached: Option<&NetOutputs>,
    keep: bool,
    progress: &Progress,
) -> Result<(Image, Option<NetOutputs>), Error> {
    let reuse = Reuse {
        nets: cached,
        response: None,
    };
    let keep = Keep {
        nets: keep,
        response: false,
    };
    process_with(img, params, reuse, keep, progress).map(|(out, fresh)| (out, fresh.nets))
}

fn bits(img: &Image) -> Vec<u64> {
    img.data.iter().map(|v| v.to_bits()).collect()
}

/// Every model, and both Hybrid modes (parity with chunks small enough to have seams).
fn configs() -> Vec<Params> {
    let p = Params {
        psf_fwhm: 3.0,
        ..Params::default()
    };
    vec![
        Params {
            model: ModelKind::Psf,
            ..p.clone()
        },
        Params {
            model: ModelKind::AstroClean,
            ..p.clone()
        },
        Params {
            model: ModelKind::Hybrid,
            mode: Mode::Seamless,
            ..p.clone()
        },
        Params {
            model: ModelKind::Hybrid,
            mode: Mode::Parity,
            chunk_size: 60,
            ..p.clone()
        },
        Params {
            model: ModelKind::DualPsf,
            star_psf_fwhm: 2.0,
            ..p.clone()
        },
        Params {
            model: ModelKind::DualPsf,
            star_psf_fwhm: 2.0,
            mode: Mode::Parity,
            chunk_size: 60,
            ..p.clone()
        },
        Params {
            model: ModelKind::DualPsf,
            ..p.clone()
        },
        Params {
            model: ModelKind::StarMask,
            mode: Mode::Parity,
            ..p
        },
    ]
}

fn nets(img: &Image, params: &Params) -> NetOutputs {
    let (_, nets) = with_nets(img, params, None, true, &Progress::new()).unwrap();
    nets.expect("newly computed outputs are returned")
}

#[test]
fn reused_outputs_give_the_same_result_as_processing() {
    for image in ["T1.tif", "CT1.tif"] {
        let img = sample(image);
        for base in configs() {
            let first = Params {
                aggressiveness: 0.4,
                ..base.clone()
            };
            let (out, nets) = with_nets(&img, &first, None, true, &Progress::new()).unwrap();
            let nets = nets.expect("newly computed outputs are returned");
            let expected = process(&img, &first, &Progress::new()).unwrap();
            assert_eq!(bits(&out), bits(&expected), "{image} {first:?}");
            for a in [0.0, 0.25, 0.9, 1.0] {
                let p = Params {
                    aggressiveness: a,
                    ..base.clone()
                };
                let progress = Progress::new();
                let (reused, fresh) = with_nets(&img, &p, Some(&nets), true, &progress).unwrap();
                assert!(fresh.is_none(), "matching outputs are reused");
                assert_eq!(progress.fraction(), 1.0);
                let expected = process(&img, &p, &Progress::new()).unwrap();
                assert_eq!(bits(&reused), bits(&expected), "{image} {p:?}");
            }
        }
    }
}

#[test]
fn single_model_outputs_are_reused_across_modes() {
    // For PSF and AstroClean the mode only changes how colour output is rounded.
    let img = sample("CT1.tif");
    for model in [ModelKind::Psf, ModelKind::AstroClean] {
        let seamless = Params {
            model,
            mode: Mode::Seamless,
            ..Params::default()
        };
        let nets = nets(&img, &seamless);
        let parity = Params {
            mode: Mode::Parity,
            aggressiveness: 0.8,
            ..seamless
        };
        let (out, fresh) = with_nets(&img, &parity, Some(&nets), true, &Progress::new()).unwrap();
        assert!(fresh.is_none());
        let expected = process(&img, &parity, &Progress::new()).unwrap();
        assert_eq!(bits(&out), bits(&expected), "{model:?}");
    }
}

#[test]
fn outputs_match_only_the_same_network_settings() {
    let img = sample("T1.tif");
    let hybrid = Params {
        model: ModelKind::Hybrid,
        mode: Mode::Seamless,
        ..Params::default()
    };
    let seamless = nets(&img, &hybrid);
    let ok = |p: Params| seamless.matches(&img, &p);
    assert!(ok(Params {
        aggressiveness: 0.1,
        ..hybrid.clone()
    }));
    assert!(
        ok(Params {
            chunk_size: 999,
            ..hybrid.clone()
        }),
        "chunk size is parity-only"
    );
    assert!(
        ok(Params {
            psf_fwhm: 3.0 + 1e-9,
            ..hybrid.clone()
        }),
        "same PSF model"
    );
    assert!(!ok(Params {
        psf_fwhm: 3.25,
        ..hybrid.clone()
    }));
    assert!(!ok(Params {
        mode: Mode::Parity,
        ..hybrid.clone()
    }));
    assert!(!ok(Params {
        model: ModelKind::Psf,
        ..hybrid.clone()
    }));

    // A different size, or a colour image of the same size, needs new outputs.
    let smaller = img.crop(0..100, 0..100);
    assert!(!seamless.matches(&smaller, &hybrid));
    let colour = Image::new(
        img.width,
        img.height,
        3,
        img.data.iter().flat_map(|&v| [v, v, v]).collect(),
    );
    assert!(!seamless.matches(&colour, &hybrid));

    // AstroClean ignores the PSF and the mode; parity Hybrid depends on the chunk size.
    let clean = Params {
        model: ModelKind::AstroClean,
        ..Params::default()
    };
    let cleaned = nets(&img, &clean);
    assert!(cleaned.matches(
        &img,
        &Params {
            psf_fwhm: 5.0,
            mode: Mode::Parity,
            ..clean
        }
    ));
    let parity = Params {
        mode: Mode::Parity,
        chunk_size: 60,
        ..hybrid
    };
    let chunked = nets(&img, &parity);
    assert!(chunked.matches(
        &img,
        &Params {
            aggressiveness: 0.9,
            ..parity.clone()
        }
    ));
    assert!(!chunked.matches(
        &img,
        &Params {
            chunk_size: 61,
            ..parity
        }
    ));
}

#[test]
fn mismatched_outputs_are_recomputed() {
    let img = sample("T1.tif");
    let psf3 = Params {
        model: ModelKind::Psf,
        psf_fwhm: 3.0,
        ..Params::default()
    };
    let stale = nets(&img, &psf3);
    let psf4 = Params {
        psf_fwhm: 4.0,
        ..psf3
    };
    let (out, fresh) = with_nets(&img, &psf4, Some(&stale), true, &Progress::new()).unwrap();
    assert!(fresh.is_some_and(|n| n.matches(&img, &psf4)));
    assert_eq!(
        bits(&out),
        bits(&process(&img, &psf4, &Progress::new()).unwrap())
    );
}

#[test]
fn output_size_is_predicted_and_keep_is_honoured() {
    for image in ["T1.tif", "CT1.tif"] {
        let img = sample(image);
        for p in configs() {
            let nets = nets(&img, &p);
            assert_eq!(
                nets.bytes(),
                net_outputs_bytes(img.width, img.height, &p),
                "{image} {p:?}"
            );
            let (out, kept) = with_nets(&img, &p, None, false, &Progress::new()).unwrap();
            assert!(kept.is_none(), "keep = false returns no outputs");
            let expected = process(&img, &p, &Progress::new()).unwrap();
            assert_eq!(bits(&out), bits(&expected));
        }
    }
}
