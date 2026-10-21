use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use astrosharp_core::{
    BitDepth, FileFormat, FitsHeader, MaskParams, Mode, ModelKind, Params, Progress, SaveOptions,
    VERSION, process, read_image, save_image,
};
use clap::{Parser, ValueEnum};
use tracing::field::Empty;

/// Sharpen and denoise astronomical images with the AstroSharp models.
///
/// Meant as a final touch-up on stretched (non-linear) images; the models were trained
/// on stretched data.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Input image: TIFF (grey or RGB; 8/16-bit integer or 32-bit float) or FITS
    /// (.fit/.fits/.fts).
    input: PathBuf,
    /// Output file; the format follows the extension (.tif/.tiff, .fit/.fits/.fts, .png,
    /// .jpg/.jpeg). FITS output keeps a FITS input's header and adds a HISTORY line.
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = Model::DualPsf)]
    model: Model,
    /// PSF FWHM in pixels (1–8, steps of 0.25). Used by the psf, hybrid and dual-psf models
    /// (for dual-psf, away from stars).
    #[arg(long, default_value_t = 3.0)]
    psf: f64,
    /// PSF FWHM in pixels for the stars (dual-psf).
    #[arg(long, default_value_t = 3.0)]
    star_psf: f64,
    /// Blend between model output (1) and input (0).
    #[arg(long, default_value_t = 0.66)]
    aggressiveness: f64,
    /// `parity` reproduces the R app's output exactly: chunked hybrid, dual-psf with a star
    /// mask per chunk, colour rounded to 5 decimals, and star-mask of a colour image as grey
    /// RGB. `seamless` avoids the chunk seams and uses one star mask for the whole image.
    #[arg(long, value_enum, default_value_t = CliMode::Seamless)]
    mode: CliMode,
    /// Chunk size for hybrid and dual-psf in parity mode.
    #[arg(long, default_value_t = 325)]
    chunk_size: usize,
    /// Percentile of the star response above which pixels are stars (0–100).
    #[arg(
        long,
        default_value_t = 98.0,
        help_heading = "Star mask (dual-psf, star-mask)"
    )]
    sensitivity: f64,
    /// Leaves out the regions numbered up to this, scanning down each column
    /// from the left (region 0 is normally the background).
    #[arg(
        long,
        default_value_t = 2.0,
        help_heading = "Star mask (dual-psf, star-mask)"
    )]
    fix_mask: f64,
    /// Blur sigma of the mask edges in pixels (below 0.1, none).
    #[arg(
        long,
        default_value_t = 5.0,
        help_heading = "Star mask (dual-psf, star-mask)"
    )]
    feathering: f64,
    /// First shape parameter of the beta curve applied to the feathered mask.
    #[arg(
        long,
        default_value_t = 9.0,
        help_heading = "Star mask (dual-psf, star-mask)"
    )]
    blend: f64,
    /// Second shape parameter of the beta curve.
    #[arg(
        long,
        default_value_t = 100.0,
        help_heading = "Star mask (dual-psf, star-mask)"
    )]
    blend2: f64,
    /// Bits per sample: TIFF 8, 16 (default) or 32 (float); FITS 16 or 32 (default: the
    /// input FITS's depth, else 32); PNG 8 or 16 (default); JPEG 8.
    #[arg(long)]
    depth: Option<u8>,
    /// JPEG quality (1–100).
    #[arg(long, default_value_t = 90, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,
}

#[derive(Clone, Copy, ValueEnum)]
enum Model {
    /// PSF sharpening.
    Psf,
    /// AstroClean denoising.
    Astroclean,
    /// PSF sharpening, then AstroClean.
    Hybrid,
    /// Sharpening with one PSF on the stars (--star-psf) and another elsewhere (--psf),
    /// mixed through the star mask.
    DualPsf,
    /// Outputs the star mask (grey).
    StarMask,
}

#[derive(Clone, Copy, ValueEnum)]
enum CliMode {
    Seamless,
    Parity,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let params = Params {
        model: match args.model {
            Model::Psf => ModelKind::Psf,
            Model::Astroclean => ModelKind::AstroClean,
            Model::Hybrid => ModelKind::Hybrid,
            Model::DualPsf => ModelKind::DualPsf,
            Model::StarMask => ModelKind::StarMask,
        },
        psf_fwhm: args.psf,
        star_psf_fwhm: args.star_psf,
        aggressiveness: args.aggressiveness,
        mode: match args.mode {
            CliMode::Seamless => Mode::Seamless,
            CliMode::Parity => Mode::Parity,
        },
        chunk_size: args.chunk_size,
        mask: MaskParams {
            sensitivity: args.sensitivity,
            fix_mask: args.fix_mask,
            feathering: args.feathering,
            blend: args.blend,
            blend2: args.blend2,
        },
    };
    let _telemetry = astrosharp_telemetry::init("astrosharp-cli");
    let span = tracing::info_span!(
        "run",
        input = %args.input.display(),
        output = %args.output.display(),
        params = %params.describe(),
        width = Empty,
        height = Empty,
        color = Empty,
        linear = Empty,
        otel.status_description = Empty,
    );
    let result = span.in_scope(|| run(&args, &params));
    if let Err(e) = &result {
        astrosharp_telemetry::record_error(&span, &format_args!("{e:#}"));
    }
    result
}

fn run(args: &Args, params: &Params) -> Result<()> {
    let format = FileFormat::from_path(&args.output).ok_or_else(|| {
        anyhow!(
            "unsupported output file type; use .tif, .tiff, .fit, .fits, .fts, .png, .jpg or .jpeg"
        )
    })?;
    let depth = match args.depth {
        None => None,
        Some(8) => Some(BitDepth::U8),
        Some(16) => Some(BitDepth::U16),
        Some(32) => Some(BitDepth::F32),
        Some(d) => bail!("--depth must be 8, 16 or 32, not {d}"),
    };
    if let Some(d) = depth
        && !format.depths().contains(&d)
    {
        bail!("{} files cannot store {} samples", format.name(), d.name());
    }

    let t0 = Instant::now();
    let (img, fits) =
        read_image(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    let linear = img.looks_linear();
    tracing::Span::current()
        .record("width", img.width)
        .record("height", img.height)
        .record("color", img.is_color())
        .record("linear", linear);
    if linear {
        eprintln!(
            "note: {} looks linear (unstretched). AstroSharp is meant for stretched images; \
             stretch it first for best results.",
            args.input.display()
        );
    }
    let t1 = Instant::now();
    let out = process(&img, params, &Progress::new())?;
    let t2 = Instant::now();
    // FITS output keeps a FITS input's sample type unless --depth says otherwise.
    let source_depth = fits
        .as_ref()
        .filter(|_| format == FileFormat::Fits)
        .and_then(FitsHeader::source_depth);
    let save = SaveOptions {
        format,
        depth: depth.or(source_depth).unwrap_or(format.default_depth()),
        jpeg_quality: args.quality,
    };
    let mut header = fits.unwrap_or_default();
    header.add_history(&format!("AstroSharp {VERSION}: {}", params.describe()));
    save_image(&args.output, &out, &save, Some(&header))
        .with_context(|| format!("writing {}", args.output.display()))?;
    let t3 = Instant::now();

    eprintln!(
        "{}x{} {} | read {:.2}s, process {:.2}s ({:.1} MP/s), write {:.2}s",
        img.width,
        img.height,
        if img.is_color() { "RGB" } else { "grey" },
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        (img.width * img.height) as f64 / 1e6 / (t2 - t1).as_secs_f64(),
        (t3 - t2).as_secs_f64(),
    );
    Ok(())
}
