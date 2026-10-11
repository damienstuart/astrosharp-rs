use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use astrosharp_core::{
    BitDepth, FileFormat, Mode, ModelKind, Params, Progress, SaveOptions, process, read_tiff,
    save_image,
};
use clap::{Parser, ValueEnum};

/// Sharpen and denoise astronomical images with the AstroSharp models.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Input TIFF (grey or RGB; 8/16-bit integer or 32-bit float).
    input: PathBuf,
    /// Output file; the format follows the extension (.tif/.tiff, .png, .jpg/.jpeg).
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = Model::Hybrid)]
    model: Model,
    /// PSF FWHM in pixels (1–8, steps of 0.25). Used by the psf and hybrid models.
    #[arg(long, default_value_t = 3.0)]
    psf: f64,
    /// Blend between model output (1) and input (0).
    #[arg(long, default_value_t = 0.66)]
    aggressiveness: f64,
    /// `parity` reproduces the R app's chunked Hybrid output exactly; `seamless` avoids its seams.
    #[arg(long, value_enum, default_value_t = CliMode::Seamless)]
    mode: CliMode,
    /// Chunk size for hybrid in parity mode.
    #[arg(long, default_value_t = 325)]
    chunk_size: usize,
    /// Bits per sample: TIFF 8, 16 (default) or 32 (float); PNG 8 or 16 (default); JPEG 8.
    #[arg(long)]
    depth: Option<u8>,
    /// JPEG quality (1–100).
    #[arg(long, default_value_t = 90, value_parser = clap::value_parser!(u8).range(1..=100))]
    quality: u8,
}

#[derive(Clone, Copy, ValueEnum)]
enum Model {
    Psf,
    Astroclean,
    Hybrid,
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
        },
        psf_fwhm: args.psf,
        aggressiveness: args.aggressiveness,
        mode: match args.mode {
            CliMode::Seamless => Mode::Seamless,
            CliMode::Parity => Mode::Parity,
        },
        chunk_size: args.chunk_size,
    };

    let format = FileFormat::from_path(&args.output).ok_or_else(|| {
        anyhow!("unsupported output file type; use .tif, .tiff, .png, .jpg or .jpeg")
    })?;
    let depth = match args.depth {
        None => format.default_depth(),
        Some(8) => BitDepth::U8,
        Some(16) => BitDepth::U16,
        Some(32) => BitDepth::F32,
        Some(d) => bail!("--depth must be 8, 16 or 32, not {d}"),
    };
    if !format.depths().contains(&depth) {
        bail!(
            "{} files cannot store {} samples",
            format.name(),
            depth.name()
        );
    }
    let save = SaveOptions {
        format,
        depth,
        jpeg_quality: args.quality,
    };

    let t0 = Instant::now();
    let img =
        read_tiff(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    let t1 = Instant::now();
    let out = process(&img, &params, &Progress::new())?;
    let t2 = Instant::now();
    save_image(&args.output, &out, &save)
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
