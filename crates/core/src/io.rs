//! Image I/O. TIFF reading/writing uses the same value scaling as R's `tiff` package;
//! FITS is handled by [`crate::fits`]; processed images can also be saved as PNG or JPEG.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::encoder::{TiffEncoder, colortype};
use tiff::tags::Tag;

use crate::fits::{self, FitsHeader};
use crate::{Error, Image};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BitDepth {
    /// 8-bit unsigned integer.
    U8,
    /// 16-bit unsigned integer, as written by the R app.
    #[default]
    U16,
    /// 32-bit IEEE float.
    F32,
}

impl BitDepth {
    pub fn name(self) -> &'static str {
        match self {
            BitDepth::U8 => "8-bit",
            BitDepth::U16 => "16-bit",
            BitDepth::F32 => "32-bit float",
        }
    }
}

/// File formats processed images can be saved in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FileFormat {
    #[default]
    Tiff,
    Fits,
    Png,
    Jpeg,
}

impl FileFormat {
    pub const ALL: [FileFormat; 4] = [
        FileFormat::Tiff,
        FileFormat::Fits,
        FileFormat::Png,
        FileFormat::Jpeg,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FileFormat::Tiff => "TIFF",
            FileFormat::Fits => "FITS",
            FileFormat::Png => "PNG",
            FileFormat::Jpeg => "JPEG",
        }
    }

    /// File extensions, preferred first.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            FileFormat::Tiff => &["tif", "tiff"],
            FileFormat::Fits => &["fit", "fits", "fts"],
            FileFormat::Png => &["png"],
            FileFormat::Jpeg => &["jpg", "jpeg"],
        }
    }

    /// The format matching `path`'s extension (case-insensitive), if any.
    pub fn from_path(path: impl AsRef<Path>) -> Option<FileFormat> {
        let ext = path.as_ref().extension()?.to_str()?.to_ascii_lowercase();
        FileFormat::ALL
            .into_iter()
            .find(|f| f.extensions().contains(&ext.as_str()))
    }

    /// Bit depths the format can store.
    pub fn depths(self) -> &'static [BitDepth] {
        match self {
            FileFormat::Tiff => &[BitDepth::U8, BitDepth::U16, BitDepth::F32],
            FileFormat::Fits => &[BitDepth::U16, BitDepth::F32],
            FileFormat::Png => &[BitDepth::U8, BitDepth::U16],
            FileFormat::Jpeg => &[BitDepth::U8],
        }
    }

    /// 16 bits where the format allows it (as the R app wrote), 32-bit float for FITS
    /// (Siril's working format); JPEG is always 8-bit. A FITS source's own depth takes
    /// precedence when saving FITS (see [`FitsHeader::source_depth`]).
    pub fn default_depth(self) -> BitDepth {
        match self {
            FileFormat::Tiff | FileFormat::Png => BitDepth::U16,
            FileFormat::Fits => BitDepth::F32,
            FileFormat::Jpeg => BitDepth::U8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveOptions {
    pub format: FileFormat,
    pub depth: BitDepth,
    /// JPEG quality, 1–100.
    pub jpeg_quality: u8,
}

impl Default for SaveOptions {
    fn default() -> Self {
        SaveOptions {
            format: FileFormat::Tiff,
            depth: BitDepth::U16,
            jpeg_quality: 90,
        }
    }
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

/// Reads a TIFF or FITS image (chosen by extension). FITS files also return their
/// header, so it can be preserved when saving.
#[tracing::instrument(
    skip_all,
    fields(path = %path.as_ref().display(), format, width, height, channels)
)]
pub fn read_image(path: impl AsRef<Path>) -> Result<(Image, Option<FitsHeader>), Error> {
    let (img, header) = if FileFormat::from_path(&path) == Some(FileFormat::Fits) {
        let (img, header) = fits::read_fits(path)?;
        (img, Some(header))
    } else {
        (read_tiff(path)?, None)
    };
    let format = if header.is_some() {
        FileFormat::Fits
    } else {
        FileFormat::Tiff
    };
    tracing::Span::current()
        .record("format", format.name())
        .record("width", img.width)
        .record("height", img.height)
        .record("channels", img.channels);
    Ok((img, header))
}

/// Saves `img` (values clipped to [0, 1]) in the format and depth given by `opts`.
/// `fits` supplies the header (and extensions) to preserve when writing FITS.
#[tracing::instrument(
    skip_all,
    fields(
        path = %path.as_ref().display(),
        format = opts.format.name(),
        depth = opts.depth.name(),
        jpeg_quality = (opts.format == FileFormat::Jpeg).then_some(opts.jpeg_quality),
    )
)]
pub fn save_image(
    path: impl AsRef<Path>,
    img: &Image,
    opts: &SaveOptions,
    fits: Option<&FitsHeader>,
) -> Result<(), Error> {
    if !opts.format.depths().contains(&opts.depth) {
        return Err(Error::Unsupported(format!(
            "{} files cannot store {} samples",
            opts.format.name(),
            opts.depth.name()
        )));
    }
    match opts.format {
        FileFormat::Tiff => write_tiff(path, img, opts.depth),
        FileFormat::Fits => fits::write_fits(path, img, opts.depth, fits),
        FileFormat::Png => write_png(path, img, opts.depth),
        FileFormat::Jpeg => write_jpeg(path, img, opts.jpeg_quality),
    }
}

fn clip(v: f64) -> f64 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

/// 16-bit samples truncate `v * 65535`, like `writeTIFF(..., bits.per.sample = 16)`.
fn to_u16(v: f64) -> u16 {
    (clip(v) * 65535.0) as u16
}

/// 8-bit samples round to the nearest level.
fn to_u8(v: f64) -> u8 {
    (clip(v) * 255.0).round() as u8
}

/// Writes an uncompressed grey or RGB TIFF. Values are clipped to [0, 1].
pub fn write_tiff(path: impl AsRef<Path>, img: &Image, depth: BitDepth) -> Result<(), Error> {
    let mut enc = TiffEncoder::new(BufWriter::new(File::create(path)?))?;
    let (w, h) = (img.width as u32, img.height as u32);
    match depth {
        BitDepth::U8 => {
            let data: Vec<u8> = img.data.iter().map(|&v| to_u8(v)).collect();
            if img.is_color() {
                enc.write_image::<colortype::RGB8>(w, h, &data)?;
            } else {
                enc.write_image::<colortype::Gray8>(w, h, &data)?;
            }
        }
        BitDepth::U16 => {
            let data: Vec<u16> = img.data.iter().map(|&v| to_u16(v)).collect();
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

/// Writes a lossless 8- or 16-bit grey or RGB PNG.
fn write_png(path: impl AsRef<Path>, img: &Image, depth: BitDepth) -> Result<(), Error> {
    let file = BufWriter::new(File::create(path)?);
    let mut enc = png::Encoder::new(file, img.width as u32, img.height as u32);
    enc.set_color(if img.is_color() {
        png::ColorType::Rgb
    } else {
        png::ColorType::Grayscale
    });
    let data: Vec<u8> = match depth {
        BitDepth::U8 => {
            enc.set_depth(png::BitDepth::Eight);
            img.data.iter().map(|&v| to_u8(v)).collect()
        }
        BitDepth::U16 => {
            enc.set_depth(png::BitDepth::Sixteen);
            img.data
                .iter()
                .flat_map(|&v| to_u16(v).to_be_bytes())
                .collect()
        }
        BitDepth::F32 => return Err(Error::Unsupported("32-bit float PNG".into())),
    };
    let mut writer = enc.write_header()?;
    writer.write_image_data(&data)?;
    writer.finish()?;
    Ok(())
}

/// Writes an 8-bit grey or RGB JPEG without chroma subsampling (keeps small coloured
/// stars crisp).
fn write_jpeg(path: impl AsRef<Path>, img: &Image, quality: u8) -> Result<(), Error> {
    let side = |n: usize| {
        u16::try_from(n)
            .map_err(|_| Error::Unsupported("JPEG is limited to 65535 pixels per side".into()))
    };
    let (w, h) = (side(img.width)?, side(img.height)?);
    let data: Vec<u8> = img.data.iter().map(|&v| to_u8(v)).collect();
    let mut enc = jpeg_encoder::Encoder::new_file(path, quality.clamp(1, 100))?;
    enc.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_4_4);
    let color = if img.is_color() {
        jpeg_encoder::ColorType::Rgb
    } else {
        jpeg_encoder::ColorType::Luma
    };
    enc.encode(&data, w, h, color)?;
    Ok(())
}
