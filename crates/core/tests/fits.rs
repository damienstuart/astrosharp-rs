//! FITS reading/writing against hand-built files.

use std::path::PathBuf;

use astrosharp_core::fits::{read_fits, write_fits};
use astrosharp_core::{BitDepth, Error, FileFormat, FitsHeader, Image, SaveOptions, save_image};

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("astrosharp-fits-{}-{name}", std::process::id()))
}

/// Assembles a FITS file: header cards, data (padded to 2880 bytes), then `tail`.
fn raw_fits(cards: &[&str], data: &[u8], tail: &[u8]) -> Vec<u8> {
    let mut b: Vec<u8> = cards
        .iter()
        .chain(&["END"])
        .flat_map(|c| format!("{c:<80}").into_bytes())
        .collect();
    b.resize(b.len().div_ceil(2880) * 2880, b' ');
    let start = b.len();
    b.extend_from_slice(data);
    b.resize(start + data.len().div_ceil(2880) * 2880, 0);
    b.extend_from_slice(tail);
    b
}

fn write_raw(name: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_path(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn header_cards(bytes: &[u8]) -> Vec<String> {
    bytes
        .chunks(80)
        .map(|c| String::from_utf8_lossy(c).into_owned())
        .take_while(|c| !c.starts_with("END "))
        .collect()
}

fn f32_be(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_be_bytes()).collect()
}

#[test]
fn reads_bottom_up_unsigned_16bit_grey() {
    // File rows are bottom-up: first row stored is the bottom of the image.
    let stored: [u16; 6] = [0, 1000, 65535, 10, 20, 30];
    let data: Vec<u8> = stored
        .iter()
        .flat_map(|&u| ((u as i32 - 32768) as i16).to_be_bytes())
        .collect();
    let path = write_raw(
        "u16.fit",
        &raw_fits(
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                   16",
                "NAXIS   =                    2",
                "NAXIS1  =                    3",
                "NAXIS2  =                    2",
                "BZERO   =                32768",
                "BSCALE  =                    1",
            ],
            &data,
            &[],
        ),
    );
    let (img, header) = read_fits(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let expected: Vec<f64> = [10.0, 20.0, 30.0, 0.0, 1000.0, 65535.0]
        .iter()
        .map(|v| v / 65535.0)
        .collect();
    assert_eq!((img.width, img.height, img.channels), (3, 2, 1));
    assert_eq!(img.data, expected);
    assert_eq!(header.source_depth(), Some(BitDepth::U16));
}

#[test]
fn reads_top_down_float_rgb_planes() {
    // 2x2 RGB, planes R, G, B; ROWORDER says the first stored row is the top.
    let r = [0.1, 0.2, 0.3, 0.4];
    let g = [0.5, 0.6, 0.7, 0.8];
    let b = [0.9, 1.0, 0.0, 0.25];
    let data = f32_be(&[r, g, b].concat());
    let path = write_raw(
        "rgb.fit",
        &raw_fits(
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    3",
                "NAXIS1  =                    2",
                "NAXIS2  =                    2",
                "NAXIS3  =                    3",
                "ROWORDER= 'TOP-DOWN'",
            ],
            &data,
            &[],
        ),
    );
    let (img, header) = read_fits(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!((img.width, img.height, img.channels), (2, 2, 3));
    let expected: Vec<f64> = (0..4)
        .flat_map(|i| [r[i], g[i], b[i]])
        .map(|v: f32| v as f64)
        .collect();
    assert_eq!(img.data, expected);
    assert_eq!(header.source_depth(), Some(BitDepth::F32));
}

#[test]
fn round_trip_preserves_header_rows_and_extensions() {
    let pixels: Vec<f32> = (0..12).map(|i| i as f32 / 16.0).collect();
    let extension = raw_fits(
        &[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "PCOUNT  =                    0",
            "GCOUNT  =                    1",
        ],
        &[],
        &[],
    );
    let source = raw_fits(
        &[
            "SIMPLE  =                    T / conforms",
            "BITPIX  =                  -32",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    3",
            "EXTEND  =                    T",
            "COMMENT   FITS format is defined in 'Astronomy and Astrophysics'",
            "BZERO   =                   0. / Offset",
            "BSCALE  =                   1. / Default scaling factor",
            "DATE    = '2020-01-01T00:00:00' / UTC date that FITS file was created",
            "OBJECT  = 'NGC 2244'           / Name of the object of interest",
            "CRPIX1  =               2220.5 / Axis1 reference pixel",
            "A_2_0   = 8.88770945070578E-09",
            "LONGSTR = 'abc&'",
            "CONTINUE  'def'",
            "CHECKSUM= 'stale'",
            "DATAMAX =                   1.",
            "HISTORY Plate Solve",
        ],
        &f32_be(&pixels),
        &extension,
    );
    let src_path = write_raw("src.fit", &source);
    let (img, mut header) = read_fits(&src_path).unwrap();
    assert_eq!(header.value("OBJECT").as_deref(), Some("NGC 2244"));
    header.add_history("AstroSharp test: Hybrid (PSF 3, aggressiveness 0.66, seamless)");

    let out_path = temp_path("out.fit");
    write_fits(&out_path, &img, BitDepth::F32, Some(&header)).unwrap();
    let out = std::fs::read(&out_path).unwrap();
    let (back, _) = read_fits(&out_path).unwrap();
    std::fs::remove_file(&src_path).unwrap();
    std::fs::remove_file(&out_path).unwrap();

    assert_eq!(out.len() % 2880, 0);
    assert_eq!(back, img);
    let cards = header_cards(&out);
    let keys: Vec<&str> = cards.iter().map(|c| c[..8].trim_end()).collect();
    assert_eq!(
        keys,
        [
            "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "COMMENT", "DATE", "OBJECT",
            "CRPIX1", "A_2_0", "LONGSTR", "CONTINUE", "HISTORY", "HISTORY",
        ]
    );
    // Preserved cards are byte-for-byte identical.
    for card in [
        "OBJECT  = 'NGC 2244'           / Name of the object of interest",
        "A_2_0   = 8.88770945070578E-09",
        "CONTINUE  'def'",
        "HISTORY Plate Solve",
    ] {
        assert!(cards.contains(&format!("{card:<80}")), "{card}");
    }
    assert!(
        !cards.iter().any(|c| c.contains("2020-01-01")),
        "DATE was not refreshed"
    );
    assert!(
        cards
            .last()
            .unwrap()
            .starts_with("HISTORY AstroSharp test: Hybrid")
    );
    // Same row order and values as the source, and the extension HDU is untouched.
    let data_start = ((cards.len() + 1) * 80).div_ceil(2880) * 2880; // cards + END
    assert_eq!(&out[data_start..data_start + 48], &f32_be(&pixels)[..]);
    assert!(out.ends_with(&extension));
}

#[test]
fn writes_minimal_header_for_non_fits_sources() {
    let img = Image::new(3, 2, 1, vec![0.0, 0.5, 1.0, 0.25, 0.75, 2.0]);
    let path = temp_path("plain.fit");
    let opts = SaveOptions {
        format: FileFormat::Fits,
        depth: BitDepth::U16,
        ..SaveOptions::default()
    };
    save_image(&path, &img, &opts, None).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let (back, header) = read_fits(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let cards = header_cards(&bytes);
    assert!(
        cards
            .iter()
            .any(|c| c.starts_with("BZERO   =                32768"))
    );
    assert_eq!(header.value("ROWORDER").as_deref(), Some("BOTTOM-UP"));
    let expected: Vec<f64> = img
        .data
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 65535.0) as u16 as f64 / 65535.0)
        .collect();
    assert_eq!(back.data, expected);
}

#[test]
fn unnormalised_float_data_is_restored_on_write() {
    let pixels: Vec<f32> = vec![0.0, 1200.0, 40000.0, 65535.0];
    let src = write_raw(
        "adu.fit",
        &raw_fits(
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
                "NAXIS2  =                    2",
            ],
            &f32_be(&pixels),
            &[],
        ),
    );
    let (img, header) = read_fits(&src).unwrap();
    assert!(img.data.iter().all(|v| (0.0..=1.0).contains(v)));
    let out = temp_path("adu-out.fit");
    write_fits(&out, &img, BitDepth::F32, Some(&header)).unwrap();
    let bytes = std::fs::read(&out).unwrap();
    std::fs::remove_file(&src).unwrap();
    std::fs::remove_file(&out).unwrap();
    let back: Vec<f32> = bytes[2880..2880 + 16]
        .chunks(4)
        .map(|c| f32::from_be_bytes(c.try_into().unwrap()))
        .collect();
    for (a, b) in back.iter().zip(&pixels) {
        assert!((a - b).abs() < 1e-2, "{a} vs {b}");
    }
}

#[test]
fn rejects_unsupported_files() {
    let two_planes = write_raw(
        "planes.fit",
        &raw_fits(
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    3",
                "NAXIS1  =                    1",
                "NAXIS2  =                    1",
                "NAXIS3  =                    2",
            ],
            &f32_be(&[0.0, 0.0]),
            &[],
        ),
    );
    let truncated = write_raw("short.fit", &{
        let mut b = raw_fits(
            &[
                "SIMPLE  =                    T",
                "BITPIX  =                  -32",
                "NAXIS   =                    2",
                "NAXIS1  =                  100",
                "NAXIS2  =                  100",
            ],
            &[],
            &[],
        );
        b.truncate(2880 + 100);
        b
    });
    for path in [&two_planes, &truncated] {
        assert!(
            matches!(read_fits(path), Err(Error::Unsupported(_))),
            "{}",
            path.display()
        );
        std::fs::remove_file(path).unwrap();
    }
    let opts = SaveOptions {
        format: FileFormat::Fits,
        depth: BitDepth::U8,
        ..SaveOptions::default()
    };
    let img = Image::new(1, 1, 1, vec![0.5]);
    let err = save_image(
        temp_path("u8.fit"),
        &img,
        &opts,
        Some(&FitsHeader::default()),
    );
    assert!(matches!(err, Err(Error::Unsupported(_))));
}
