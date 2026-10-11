//! The AstroSharp processing pipeline, following `First3BetasFun.R` (PSF and
//! AstroClean models) and `Combined_Sharpen_and_Clean.R` (Hybrid model).

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rayon::prelude::*;

use crate::features::{self, N_FEATURES, RADIUS};
use crate::model::{Mlp, Scratch};
use crate::{Error, Image, Plane, color, weights};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// PSF sharpening model (the app's "PSF Model (Pre-Beta)").
    Psf,
    /// AstroClean denoising model.
    AstroClean,
    /// PSF sharpening followed by AstroClean ("Hybrid Model").
    Hybrid,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Whole-image passes. Identical to `Parity` for PSF and AstroClean; for Hybrid it
    /// avoids the faint seams the R app's chunked processing leaves.
    #[default]
    Seamless,
    /// Reproduces the R app's chunked Hybrid processing exactly.
    Parity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub model: ModelKind,
    /// Selects the PSF model: 1–8 pixels in steps of 0.25.
    pub psf_fwhm: f64,
    /// Blend between the model output (1) and the input (0).
    pub aggressiveness: f64,
    pub mode: Mode,
    /// Chunk size for Hybrid in `Mode::Parity` (the app's "Processing Chunk Size").
    pub chunk_size: usize,
}

impl Default for Params {
    /// The R app's defaults.
    fn default() -> Self {
        Params {
            model: ModelKind::Psf,
            psf_fwhm: 3.0,
            aggressiveness: 0.66,
            mode: Mode::Seamless,
            chunk_size: 325,
        }
    }
}

/// Smallest width/height the 9x9 models can process.
pub const MIN_SIZE: usize = 10;
/// Smallest chunk size for which R's chunk layout (12 px of overlap) is well formed.
pub const MIN_CHUNK_SIZE: usize = 13;

/// Shared progress counter and cancellation flag for a running [`process`] call.
#[derive(Debug, Default)]
pub struct Progress {
    done: AtomicU64,
    total: AtomicU64,
    cancelled: AtomicBool,
}

impl Progress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fraction of work done, in [0, 1].
    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        (self.done.load(Ordering::Relaxed) as f64 / total as f64).min(1.0) as f32
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    fn start(&self, total: usize) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total as u64, Ordering::Relaxed);
    }

    fn advance(&self, n: usize) {
        self.done.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// Processes a grey or RGB image. Colour images are converted to CIE Luv and only the
/// lightness channel is processed, as in the R app.
pub fn process(img: &Image, params: &Params, progress: &Progress) -> Result<Image, Error> {
    if img.width < MIN_SIZE || img.height < MIN_SIZE {
        return Err(Error::InvalidParams(format!(
            "image must be at least {MIN_SIZE}x{MIN_SIZE} pixels"
        )));
    }
    let (w, h) = (img.width, img.height);
    if !img.is_color() {
        let out = process_plane(&Plane::new(w, h, img.data.clone()), params, progress)?;
        return Ok(Image::new(w, h, 1, out.data));
    }
    let luv: Vec<[f64; 3]> = img
        .data
        .par_chunks_exact(3)
        .map(|p| color::srgb_to_luv([p[0], p[1], p[2]]))
        .collect();
    let lightness = Plane::new(w, h, luv.iter().map(|p| p[0] / 100.0).collect());
    let out = process_plane(&lightness, params, progress)?;
    let data = luv
        .par_iter()
        .zip(&out.data)
        .flat_map_iter(|(p, &l)| color::luv_to_srgb([l * 100.0, p[1], p[2]]))
        .collect();
    Ok(Image::new(w, h, 3, data))
}

/// Processes one channel (grey values, or Luv lightness / 100).
pub fn process_plane(plane: &Plane, params: &Params, progress: &Progress) -> Result<Plane, Error> {
    let a = params.aggressiveness;
    if !(0.0..=1.0).contains(&a) {
        return Err(Error::InvalidParams(format!(
            "aggressiveness must be between 0 and 1, got {a}"
        )));
    }
    if plane.width < MIN_SIZE || plane.height < MIN_SIZE {
        return Err(Error::InvalidParams(format!(
            "image must be at least {MIN_SIZE}x{MIN_SIZE} pixels"
        )));
    }
    match params.model {
        ModelKind::Psf => single(weights::psf(params.psf_fwhm)?, plane, a, progress),
        ModelKind::AstroClean => single(weights::astroclean(), plane, a, progress),
        ModelKind::Hybrid => {
            let psf = weights::psf(params.psf_fwhm)?;
            let clean = weights::astroclean();
            match params.mode {
                Mode::Seamless => hybrid_seamless(psf, clean, plane, a, progress),
                Mode::Parity => {
                    if params.chunk_size < MIN_CHUNK_SIZE {
                        return Err(Error::InvalidParams(format!(
                            "chunk size must be at least {MIN_CHUNK_SIZE}"
                        )));
                    }
                    hybrid_parity(psf, clean, plane, a, params.chunk_size, progress)
                }
            }
        }
    }
}

/// The region R's preview mode processes: rows `round((x ± 0.05) * height)` and columns
/// `round((y ± 0.05) * width)` (1-based, inclusive), where `x` and `y` are the app's
/// `prevx()` / `prevy()`. Returned as 0-based half-open ranges.
pub fn r_preview_region(
    height: usize,
    width: usize,
    x: f64,
    y: f64,
) -> (Range<usize>, Range<usize>) {
    let span = |p: f64, n: usize| {
        let lo = ((p - 0.05) * n as f64).round_ties_even().max(1.0) as usize;
        let hi = ((p + 0.05) * n as f64).round_ties_even().min(n as f64) as usize;
        (lo - 1)..hi
    };
    (span(x, height), span(y, width))
}

/// Pixels the R app writes model output to: 1-based `5:(n - 5)`. The remaining border
/// (4 px at the start, 5 px at the end) keeps the input values.
fn kept(n: usize) -> Range<usize> {
    RADIUS..n - 5
}

/// Single-model processing: `a * model + (1 - a) * input` inside the kept region.
/// The R code works in chunks, but every output depends only on its own 9x9 window,
/// so a plain sliding window gives identical results.
fn single(mlp: &Mlp, plane: &Plane, a: f64, progress: &Progress) -> Result<Plane, Error> {
    let (rows, cols) = (kept(plane.height), kept(plane.width));
    progress.start(rows.len() * cols.len());
    let nn = eval_windows(mlp, plane, rows.clone(), cols.clone(), progress)?;
    let mut out = plane.clone();
    for (i, r) in rows.enumerate() {
        for (j, c) in cols.clone().enumerate() {
            let v = a * nn[i * cols.len() + j] + (1.0 - a) * plane.at(r, c);
            out.set(r, c, v);
        }
    }
    Ok(out)
}

/// Hybrid output: R's `(a * clean) + (sharp * (1 - a))`, clamped to [0, 1].
fn hybrid_blend(a: f64, clean: f64, sharp: f64) -> f64 {
    (a * clean + sharp * (1.0 - a)).clamp(0.0, 1.0)
}

fn hybrid_seamless(
    psf: &Mlp,
    clean: &Mlp,
    plane: &Plane,
    a: f64,
    progress: &Progress,
) -> Result<Plane, Error> {
    let (w, h) = (plane.width, plane.height);
    let (rows, cols) = (kept(h), kept(w));
    progress.start(w * h + rows.len() * cols.len());
    let padded = plane.padded(RADIUS);
    let sharp = eval_windows(
        psf,
        &padded,
        RADIUS..h + RADIUS,
        RADIUS..w + RADIUS,
        progress,
    )?;
    let sharp = Plane::new(w, h, sharp);
    let cleaned = eval_windows(clean, &sharp, rows.clone(), cols.clone(), progress)?;
    let mut out = Plane::new(
        w,
        h,
        plane.data.iter().map(|&x| hybrid_blend(a, x, x)).collect(),
    );
    for (i, r) in rows.enumerate() {
        for (j, c) in cols.clone().enumerate() {
            out.set(
                r,
                c,
                hybrid_blend(a, cleaned[i * cols.len() + j], sharp.at(r, c)),
            );
        }
    }
    Ok(out)
}

/// R's `seq(1, n, seg)`, with `n` appended if it is not already the last element.
fn r_breaks(n: usize, seg: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (1..=n).step_by(seg).collect();
    if v.last() != Some(&n) {
        v.push(n);
    }
    v
}

/// 0-based chunk extents along one axis, as in `MixedPSFModel`: every chunk but the
/// first starts 12 px early so the 9x9 windows cover what the previous chunk missed.
fn chunk_spans(n: usize, seg: usize) -> Vec<Range<usize>> {
    r_breaks(n, seg)
        .windows(2)
        .map(|p| {
            let start = if p[0] == 1 { 1 } else { p[0] - 12 };
            start - 1..p[1] - 1
        })
        .collect()
}

/// Hybrid exactly as the R app computes it: per chunk, PSF on the chunk interior, then
/// AstroClean on the chunk whose interior was replaced (its 4 px rim stays unsharpened),
/// with later chunks overwriting earlier ones.
fn hybrid_parity(
    psf: &Mlp,
    clean: &Mlp,
    plane: &Plane,
    a: f64,
    seg: usize,
    progress: &Progress,
) -> Result<Plane, Error> {
    let padded = plane.padded(RADIUS);
    let row_spans = chunk_spans(padded.height, seg);
    let col_spans = chunk_spans(padded.width, seg);
    let interior = |s: &Range<usize>| s.len() - 2 * RADIUS;
    let work: usize = col_spans.iter().map(interior).sum::<usize>()
        * row_spans.iter().map(interior).sum::<usize>();
    progress.start(2 * work);

    let mut sharp = Plane::zeros(padded.width, padded.height);
    let mut cleaned = Plane::zeros(padded.width, padded.height);
    for cs in &col_spans {
        for rs in &row_spans {
            let mut chunk = padded.sub(rs.clone(), cs.clone());
            let ir = RADIUS..chunk.height - RADIUS;
            let ic = RADIUS..chunk.width - RADIUS;
            let s1 = eval_windows(psf, &chunk, ir.clone(), ic.clone(), progress)?;
            chunk.write_region(ir.clone(), ic.clone(), &s1);
            let s2 = eval_windows(clean, &chunk, ir, ic, progress)?;
            let dr = rs.start + RADIUS..rs.end - RADIUS;
            let dc = cs.start + RADIUS..cs.end - RADIUS;
            sharp.write_region(dr.clone(), dc.clone(), &s1);
            cleaned.write_region(dr, dc, &s2);
        }
    }

    // R's fill_matrix + trimming maps original pixel (r, c) to padded (r + 4, c + 4).
    let (rows, cols) = (kept(plane.height), kept(plane.width));
    let mut out = Plane::new(
        plane.width,
        plane.height,
        plane.data.iter().map(|&x| hybrid_blend(a, x, x)).collect(),
    );
    for r in rows {
        for c in cols.clone() {
            let (pr, pc) = (r + RADIUS, c + RADIUS);
            out.set(r, c, hybrid_blend(a, cleaned.at(pr, pc), sharp.at(pr, pc)));
        }
    }
    Ok(out)
}

/// Pixels per batched forward pass.
const TILE: usize = 512;

/// Evaluates `mlp` on the 9x9 window around every pixel of `rows x cols` in `src`
/// (all windows must lie inside `src`). Returns the outputs row-major.
fn eval_windows(
    mlp: &Mlp,
    src: &Plane,
    rows: Range<usize>,
    cols: Range<usize>,
    progress: &Progress,
) -> Result<Vec<f64>, Error> {
    let w = cols.len();
    let mut out = vec![0.0; rows.len() * w];
    if w == 0 {
        return Ok(out);
    }
    out.par_chunks_mut(w).enumerate().try_for_each_init(
        || (Scratch::default(), Vec::new()),
        |(scratch, feats), (i, row_out)| {
            if progress.is_cancelled() {
                return Err(Error::Cancelled);
            }
            let r = rows.start + i;
            for (t, tile) in row_out.chunks_mut(TILE).enumerate() {
                let c0 = cols.start + t * TILE;
                let n = tile.len();
                feats.resize(n * N_FEATURES, 0.0);
                features::fill_row(src, r, c0..c0 + n, feats);
                mlp.forward(feats, n, scratch, tile);
            }
            progress.advance(w);
            Ok(())
        },
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaks_match_r_seq() {
        assert_eq!(r_breaks(508, 325), [1, 326, 508]);
        assert_eq!(r_breaks(508, 169), [1, 170, 339, 508]);
        assert_eq!(r_breaks(100, 500), [1, 100]);
    }

    #[test]
    fn spans_overlap_by_12() {
        assert_eq!(chunk_spans(508, 169), [0..169, 157..338, 326..507]);
    }

    #[test]
    fn preview_region_rounds_half_even() {
        // (0.5 - 0.05) * 50 = 22.5 -> 22, (0.5 + 0.05) * 50 = 27.5 -> 28
        assert_eq!(r_preview_region(50, 50, 0.5, 0.5), (21..28, 21..28));
    }
}
