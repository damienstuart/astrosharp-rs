//! TIFF reading and writing with the same value scaling as R's `tiff` package.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::encoder::{TiffEncoder, colortype};
use tiff::tags::Tag;

use crate::{Error, Image};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BitDepth {
    /// 16-bit unsigned integer, as written by the R app.
    #[default]
    U16,
    /// 32-bit IEEE float.
    F32,
}

/// Reads a grey or RGB TIFF, scaling integer samples to [0, 1] (`x / 255`,
/// `x / 65535`, ...) and passing floating point samples through, like `readTIFF()`.
/// An alpha channel, if present, is dropped.
pub fn read_tiff(path: impl AsRef<Path>) -> Result<Image, Error> {
    let file = BufReader::new(File::open(path)?);
    let mut dec = Decoder::new(file)?.with_limits(Limits::unlimited());
    let (w, h) = dec.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let (samples, channels) = match dec.colortype()? {
        ColorType::Gray(_) => (1, 1),
        ColorType::GrayA(_) => (2, 1),
        ColorType::RGB(_) => (3, 3),
        ColorType::RGBA(_) => (4, 3),
        other => return Err(Error::Unsupported(format!("TIFF colour type {other:?}"))),
    };
    if samples > 1 && dec.get_tag_u32(Tag::PlanarConfiguration).unwrap_or(1) == 2 {
        return Err(Error::Unsupported("planar (non-interleaved) TIFF".into()));
    }
    let raw: Vec<f64> = match dec.read_image()? {
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f64 / 255.0).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f64 / 65535.0).collect(),
        DecodingResult::U32(v) => v.into_iter().map(|x| x as f64 / u32::MAX as f64).collect(),
        DecodingResult::F16(v) => v.into_iter().map(|x| f64::from(f32::from(x))).collect(),
        DecodingResult::F32(v) => v.into_iter().map(f64::from).collect(),
        DecodingResult::F64(v) => v,
        _ => return Err(Error::Unsupported("TIFF sample format".into())),
    };
    if raw.len() != w * h * samples {
        return Err(Error::Unsupported("unexpected TIFF data layout".into()));
    }
    let data = if samples == channels {
        raw
    } else {
        raw.chunks_exact(samples)
            .flat_map(|px| px[..channels].iter().copied())
            .collect()
    };
    Ok(Image::new(w, h, channels, data))
}

/// Writes an uncompressed grey or RGB TIFF. Values are clipped to [0, 1]; 16-bit output
/// truncates `v * 65535` like `writeTIFF(..., bits.per.sample = 16)`.
pub fn write_tiff(path: impl AsRef<Path>, img: &Image, depth: BitDepth) -> Result<(), Error> {
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path)?))?;
    let (w, h) = (img.width as u32, img.height as u32);
    let clip = |v: f64| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    match depth {
        BitDepth::U16 => {
            let data: Vec<u16> = img
                .data
                .iter()
                .map(|&v| (clip(v) * 65535.0) as u16)
                .collect();
            if img.is_color() {
                enc.write_image::<colortype::RGB16>(w, h, &data)?;
            } else {
                enc.write_image::<colortype::Gray16>(w, h, &data)?;
            }
        }
        BitDepth::F32 => {
            let data: Vec<f32> = img.data.iter().map(|&v| clip(v) as f32).collect();
            if img.is_color() {
                enc.write_image::<colortype::RGB32Float>(w, h, &data)?;
            } else {
                enc.write_image::<colortype::Gray32Float>(w, h, &data)?;
            }
        }
    }
    Ok(())
}
