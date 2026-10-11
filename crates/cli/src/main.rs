use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use astrosharp_core::{
    BitDepth, Mode, ModelKind, Params, Progress, process, read_tiff, write_tiff,
};
use clap::{Parser, ValueEnum};

/// Sharpen and denoise astronomical images with the AstroSharp models.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Input TIFF (grey or RGB; 8/16-bit integer or 32-bit float).
    input: PathBuf,
    /// Output TIFF.
    output: PathBuf,
    #[arg(long, value_enum, default_value_t = Model::Psf)]
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
    /// Write 32-bit float samples instead of 16-bit integers.
    #[arg(long)]
    float: bool,
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

    let t0 = Instant::now();
    let img =
        read_tiff(&args.input).with_context(|| format!("reading {}", args.input.display()))?;
    let t1 = Instant::now();
    let out = process(&img, &params, &Progress::new())?;
    let t2 = Instant::now();
    let depth = if args.float {
        BitDepth::F32
    } else {
        BitDepth::U16
    };
    write_tiff(&args.output, &out, depth)
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
