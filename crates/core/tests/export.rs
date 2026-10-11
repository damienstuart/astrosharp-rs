//! Round-trip tests for saving processed images as TIFF, PNG and JPEG.

use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;

use astrosharp_core::{BitDepth, Error, FileFormat, Image, SaveOptions, read_tiff, save_image};

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("astrosharp-test-{}-{name}", std::process::id()))
}

/// Smooth test image (JPEG-friendly), with exact 0 and 1 in the corners.
fn gradient(width: usize, height: usize, channels: usize) -> Image {
    let mut data = Vec::with_capacity(width * height * channels);
    for y in 0..height {
        for x in 0..width {
            let (u, v) = (
                x as f64 / (width - 1) as f64,
                y as f64 / (height - 1) as f64,
            );
            for c in 0..channels {
                data.push(match c {
                    0 => (u + v) / 2.0,
                    1 => u,
                    _ => 1.0 - v,
                });
            }
        }
    }
    Image::new(width, height, channels, data)
}

fn save(img: &Image, name: &str, format: FileFormat, depth: BitDepth) -> PathBuf {
    let path = temp_path(name);
    let opts = SaveOptions {
        format,
        depth,
        jpeg_quality: 95,
    };
    save_image(&path, img, &opts).unwrap();
    path
}

#[test]
fn format_from_extension() {
    assert_eq!(FileFormat::from_path("a.TIF"), Some(FileFormat::Tiff));
    assert_eq!(FileFormat::from_path("a.tiff"), Some(FileFormat::Tiff));
    assert_eq!(FileFormat::from_path("dir/a.png"), Some(FileFormat::Png));
    assert_eq!(FileFormat::from_path("a.JPeG"), Some(FileFormat::Jpeg));
    assert_eq!(FileFormat::from_path("a.jpg"), Some(FileFormat::Jpeg));
    assert_eq!(FileFormat::from_path("a.bmp"), None);
    assert_eq!(FileFormat::from_path("a"), None);
}

#[test]
fn unsupported_depths_are_rejected() {
    let img = gradient(16, 16, 1);
    for (format, depth) in [
        (FileFormat::Png, BitDepth::F32),
        (FileFormat::Jpeg, BitDepth::U16),
        (FileFormat::Jpeg, BitDepth::F32),
    ] {
        let opts = SaveOptions {
            format,
            depth,
            ..SaveOptions::default()
        };
        let err = save_image(temp_path("unsupported"), &img, &opts).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)), "{format:?} {depth:?}");
    }
}

#[test]
fn png_round_trip() {
    for channels in [1, 3] {
        let img = gradient(37, 23, channels);
        for depth in [BitDepth::U8, BitDepth::U16] {
            let path = save(
                &img,
                &format!("{channels}-{depth:?}.png"),
                FileFormat::Png,
                depth,
            );
            let mut dec = png::Decoder::new(BufReader::new(File::open(&path).unwrap()));
            dec.set_transformations(png::Transformations::IDENTITY);
            let mut reader = dec.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            std::fs::remove_file(&path).unwrap();

            assert_eq!((info.width, info.height), (37, 23));
            let expected_color = if channels == 3 {
                png::ColorType::Rgb
            } else {
                png::ColorType::Grayscale
            };
            assert_eq!(info.color_type, expected_color);
            let samples: Vec<u16> = match depth {
                BitDepth::U8 => {
                    assert_eq!(info.bit_depth, png::BitDepth::Eight);
                    buf.iter().map(|&b| b as u16).collect()
                }
                _ => {
                    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
                    buf.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|&b| u16::from_be_bytes(b))
                        .collect()
                }
            };
            let expected: Vec<u16> = img
                .data
                .iter()
                .map(|&v| match depth {
                    BitDepth::U8 => (v * 255.0).round() as u16,
                    _ => (v * 65535.0) as u16,
                })
                .collect();
            assert_eq!(samples, expected, "{channels} channel(s), {depth:?}");
        }
    }
}

#[test]
fn tiff_8bit_round_trip() {
    let img = gradient(20, 10, 3);
    let path = save(&img, "8bit.tif", FileFormat::Tiff, BitDepth::U8);
    let back = read_tiff(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    let expected: Vec<f64> = img
        .data
        .iter()
        .map(|&v| (v * 255.0).round() / 255.0)
        .collect();
    assert_eq!(back.data, expected);
}

#[test]
fn jpeg_round_trip() {
    for channels in [1, 3] {
        let img = gradient(64, 48, channels);
        let path = save(
            &img,
            &format!("{channels}.jpg"),
            FileFormat::Jpeg,
            BitDepth::U8,
        );
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let mut dec = zune_jpeg::JpegDecoder::new(Cursor::new(bytes));
        let pixels = dec.decode().unwrap();
        let info = dec.info().unwrap();
        assert_eq!((info.width, info.height), (64, 48));
        // The decoder may expand grey to RGB; compare the first `channels` of each pixel.
        let out_channels = pixels.len() / (64 * 48);
        let mut total_err = 0.0;
        let mut max_err: f64 = 0.0;
        for (px, src) in pixels
            .chunks_exact(out_channels)
            .zip(img.data.chunks_exact(channels))
        {
            for c in 0..channels {
                let err = (px[c] as f64 / 255.0 - src[c]).abs();
                total_err += err;
                max_err = max_err.max(err);
            }
        }
        let mean_err = total_err / img.data.len() as f64;
        assert!(
            mean_err < 1.5 / 255.0,
            "{channels} channel(s): mean error {mean_err}"
        );
        assert!(
            max_err < 8.0 / 255.0,
            "{channels} channel(s): max error {max_err}"
        );
    }
}
