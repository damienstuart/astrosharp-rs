//! The AstroSharp processing pipeline, following `First3BetasFun.R` (PSF and
//! AstroClean models) and `Combined_Sharpen_and_Clean.R` (Hybrid model).

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rayon::prelude::*;
use tracing::{debug_span, info_span};

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
    /// Whole-image passes at full precision: Hybrid avoids the faint seams the R app's
    /// chunked processing leaves, and colour output is not rounded to 5 decimals. Grey
    /// PSF and AstroClean output is identical to `Parity`.
    #[default]
    Seamless,
    /// Reproduces the R app's output exactly (chunked Hybrid, colour rounded to 5 decimals).
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

impl Params {
    /// Short description, e.g. `Hybrid (PSF 3, aggressiveness 0.66, seamless)`.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.model != ModelKind::AstroClean {
            parts.push(format!("PSF {}", self.psf_fwhm));
        }
        parts.push(format!("aggressiveness {:.2}", self.aggressiveness));
        if self.model == ModelKind::Hybrid {
            parts.push(match self.mode {
                Mode::Seamless => "seamless".to_string(),
                Mode::Parity => format!("R parity, chunk size {}", self.chunk_size),
            });
        }
        let name = match self.model {
            ModelKind::Psf => "PSF",
            ModelKind::AstroClean => "AstroClean",
            ModelKind::Hybrid => "Hybrid",
        };
        format!("{name} ({})", parts.join(", "))
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

/// The settings that change the neural-network outputs: everything except the
/// aggressiveness, and for the single models the mode (which only affects how colour
/// output is rounded).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetKey {
    model: ModelKind,
    /// PSF model index (see [`weights::psf_index`]); `None` for AstroClean.
    psf: Option<usize>,
    /// Hybrid only.
    mode: Option<Mode>,
    /// Hybrid in parity mode only.
    chunk_size: Option<usize>,
}

impl NetKey {
    pub fn of(params: &Params) -> NetKey {
        let hybrid = params.model == ModelKind::Hybrid;
        NetKey {
            model: params.model,
            psf: (params.model != ModelKind::AstroClean)
                .then(|| weights::psf_index(params.psf_fwhm))
                .flatten(),
            mode: hybrid.then_some(params.mode),
            chunk_size: (hybrid && params.mode == Mode::Parity).then_some(params.chunk_size),
        }
    }
}

/// The neural-network outputs for one image: everything [`process`] computes except the
/// final blend, which depends only on the aggressiveness. See [`process_with`].
#[derive(Clone, Debug)]
pub struct NetOutputs {
    key: NetKey,
    width: usize,
    height: usize,
    channels: usize,
    nets: Nets,
}

impl NetOutputs {
    /// Whether these outputs fit `img` (by size and channels) and give the same network
    /// outputs as `params`.
    pub fn matches(&self, img: &Image, params: &Params) -> bool {
        self.key == NetKey::of(params)
            && (self.width, self.height, self.channels) == (img.width, img.height, img.channels)
    }

    /// The settings these outputs were computed with.
    pub fn key(&self) -> NetKey {
        self.key
    }
}

/// Network outputs for one channel, in the layout each model's blend reads.
#[derive(Clone, Debug)]
enum Nets {
    /// PSF or AstroClean output over the kept region, row-major.
    Single(Vec<f64>),
    /// Hybrid, seamless: the sharpened channel and the cleaned kept region (row-major).
    Seamless { sharp: Plane, cleaned: Vec<f64> },
    /// Hybrid, R parity: sharpened and cleaned padded planes.
    Parity { sharp: Plane, cleaned: Plane },
}

/// Processes a grey or RGB image. Colour images are converted to CIE Luv and only the
/// lightness channel is processed, as in the R app.
pub fn process(img: &Image, params: &Params, progress: &Progress) -> Result<Image, Error> {
    let _span = process_span(img, params).entered();
    run(img, params, None, false, progress).map(|(out, _)| out)
}

/// Like [`process`], but reuses `cached` network outputs when they [match](NetOutputs::matches)
/// the image and settings, so only the blend runs (for example when just the
/// aggressiveness changed). Returns the network outputs it computed, if it had to.
pub fn process_with(
    img: &Image,
    params: &Params,
    cached: Option<&NetOutputs>,
    progress: &Progress,
) -> Result<(Image, Option<NetOutputs>), Error> {
    let span = process_span(img, params);
    let _enter = span.enter();
    let cached = cached.filter(|n| n.matches(img, params));
    span.record("reused", cached.is_some());
    run(img, params, cached, true, progress)
}

/// The `process` span: image size and settings.
fn process_span(img: &Image, params: &Params) -> tracing::Span {
    info_span!(
        "process",
        width = img.width,
        height = img.height,
        color = img.is_color(),
        model = ?params.model,
        psf_fwhm = (params.model != ModelKind::AstroClean).then_some(params.psf_fwhm),
        aggressiveness = params.aggressiveness,
        mode = ?params.mode,
        chunk_size = (params.model == ModelKind::Hybrid && params.mode == Mode::Parity)
            .then_some(params.chunk_size),
        reused = tracing::field::Empty,
    )
}

/// [`process`] with optional reuse of matching network outputs. With `keep`, newly
/// computed outputs are returned; without it they are freed before the colour conversion.
fn run(
    img: &Image,
    params: &Params,
    cached: Option<&NetOutputs>,
    keep: bool,
    progress: &Progress,
) -> Result<(Image, Option<NetOutputs>), Error> {
    if img.width < MIN_SIZE || img.height < MIN_SIZE {
        return Err(Error::InvalidParams(format!(
            "image must be at least {MIN_SIZE}x{MIN_SIZE} pixels"
        )));
    }
    let (w, h) = (img.width, img.height);
    let wrap = |nets| NetOutputs {
        key: NetKey::of(params),
        width: w,
        height: h,
        channels: img.channels,
        nets,
    };
    if !img.is_color() {
        let plane = Plane::new(w, h, img.data.clone());
        let (out, nets) = plane_stage(&plane, params, cached, progress)?;
        let fresh = nets.filter(|_| keep).map(wrap);
        return Ok((Image::new(w, h, 1, out.data), fresh));
    }
    let luv: Vec<[f64; 3]> = info_span!("to_luv").in_scope(|| {
        img.data
            .par_chunks_exact(3)
            .map(|p| color::srgb_to_luv([p[0], p[1], p[2]]))
            .collect()
    });
    let lightness = Plane::new(w, h, luv.iter().map(|p| p[0] / 100.0).collect());
    let (out, nets) = plane_stage(&lightness, params, cached, progress)?;
    // Without `keep`, the network outputs are freed here, before the conversion back.
    let fresh = nets.filter(|_| keep).map(wrap);
    let data = info_span!("from_luv").in_scope(|| {
        luv.par_iter()
            .zip(&out.data)
            .flat_map_iter(|(p, &l)| {
                let luv = [l * 100.0, p[1], p[2]];
                // R rounds to 5 decimals; only parity mode reproduces that.
                match params.mode {
                    Mode::Parity => color::luv_to_srgb(luv),
                    Mode::Seamless => color::luv_to_srgb_unrounded(luv),
                }
            })
            .collect()
    });
    Ok((Image::new(w, h, 3, data), fresh))
}

/// Processes one channel (grey values, or Luv lightness / 100).
pub fn process_plane(plane: &Plane, params: &Params, progress: &Progress) -> Result<Plane, Error> {
    plane_stage(plane, params, None, progress).map(|(out, _)| out)
}

/// Validates the settings for `plane`, runs the networks unless `cached` has their
/// outputs, and blends. Returns the network outputs it computed, if any.
fn plane_stage(
    plane: &Plane,
    params: &Params,
    cached: Option<&NetOutputs>,
    progress: &Progress,
) -> Result<(Plane, Option<Nets>), Error> {
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
    match cached {
        Some(cached) => {
            // Nothing left to wait for: show the work as done.
            progress.start(1);
            progress.advance(1);
            let out = info_span!("blend").in_scope(|| blend_plane(plane, &cached.nets, a));
            Ok((out, None))
        }
        None => {
            let nets = networks_plane(plane, params, progress)?;
            let out = info_span!("blend").in_scope(|| blend_plane(plane, &nets, a));
            Ok((out, Some(nets)))
        }
    }
}

/// Runs the model(s) `params` selects on one channel.
fn networks_plane(plane: &Plane, params: &Params, progress: &Progress) -> Result<Nets, Error> {
    match params.model {
        ModelKind::Psf => {
            let psf = weights::psf(params.psf_fwhm)?;
            info_span!("psf_pass")
                .in_scope(|| single_nets(psf, plane, progress))
                .map(Nets::Single)
        }
        ModelKind::AstroClean => info_span!("astroclean_pass")
            .in_scope(|| single_nets(weights::astroclean(), plane, progress))
            .map(Nets::Single),
        ModelKind::Hybrid => {
            let psf = weights::psf(params.psf_fwhm)?;
            let clean = weights::astroclean();
            match params.mode {
                Mode::Seamless => hybrid_seamless_nets(psf, clean, plane, progress),
                Mode::Parity => {
                    if params.chunk_size < MIN_CHUNK_SIZE {
                        return Err(Error::InvalidParams(format!(
                            "chunk size must be at least {MIN_CHUNK_SIZE}"
                        )));
                    }
                    hybrid_parity_nets(psf, clean, plane, params.chunk_size, progress)
                }
            }
        }
    }
}

/// Combines the network outputs with the input `plane` for aggressiveness `a`.
fn blend_plane(plane: &Plane, nets: &Nets, a: f64) -> Plane {
    match nets {
        Nets::Single(nn) => single_blend(plane, nn, a),
        Nets::Seamless { sharp, cleaned } => hybrid_seamless_blend(plane, sharp, cleaned, a),
        Nets::Parity { sharp, cleaned } => hybrid_parity_blend(plane, sharp, cleaned, a),
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

/// Single-model network output over the kept region, row-major. The R code works in
/// chunks, but every output depends only on its own 9x9 window, so a plain sliding window
/// gives identical results.
fn single_nets(mlp: &Mlp, plane: &Plane, progress: &Progress) -> Result<Vec<f64>, Error> {
    let (rows, cols) = (kept(plane.height), kept(plane.width));
    progress.start(rows.len() * cols.len());
    eval_windows(mlp, plane, rows, cols, progress)
}

/// Single-model output: `a * model + (1 - a) * input` inside the kept region.
fn single_blend(plane: &Plane, nn: &[f64], a: f64) -> Plane {
    let (rows, cols) = (kept(plane.height), kept(plane.width));
    let mut out = plane.clone();
    for (i, r) in rows.enumerate() {
        for (j, c) in cols.clone().enumerate() {
            let v = a * nn[i * cols.len() + j] + (1.0 - a) * plane.at(r, c);
            out.set(r, c, v);
        }
    }
    out
}

/// Hybrid output: R's `(a * clean) + (sharp * (1 - a))`, clamped to [0, 1].
fn hybrid_blend(a: f64, clean: f64, sharp: f64) -> f64 {
    (a * clean + sharp * (1.0 - a)).clamp(0.0, 1.0)
}

fn hybrid_seamless_nets(
    psf: &Mlp,
    clean: &Mlp,
    plane: &Plane,
    progress: &Progress,
) -> Result<Nets, Error> {
    let (w, h) = (plane.width, plane.height);
    let (rows, cols) = (kept(h), kept(w));
    progress.start(w * h + rows.len() * cols.len());
    let padded = plane.padded(RADIUS);
    let sharp = info_span!("psf_pass").in_scope(|| {
        eval_windows(
            psf,
            &padded,
            RADIUS..h + RADIUS,
            RADIUS..w + RADIUS,
            progress,
        )
    })?;
    let sharp = Plane::new(w, h, sharp);
    let cleaned = info_span!("astroclean_pass")
        .in_scope(|| eval_windows(clean, &sharp, rows, cols, progress))?;
    Ok(Nets::Seamless { sharp, cleaned })
}

fn hybrid_seamless_blend(plane: &Plane, sharp: &Plane, cleaned: &[f64], a: f64) -> Plane {
    let (w, h) = (plane.width, plane.height);
    let (rows, cols) = (kept(h), kept(w));
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
    out
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

/// Hybrid network outputs exactly as the R app computes them: per chunk, PSF on the
/// chunk interior, then AstroClean on the chunk whose interior was replaced (its 4 px rim
/// stays unsharpened), with later chunks overwriting earlier ones.
fn hybrid_parity_nets(
    psf: &Mlp,
    clean: &Mlp,
    plane: &Plane,
    seg: usize,
    progress: &Progress,
) -> Result<Nets, Error> {
    let padded = plane.padded(RADIUS);
    let row_spans = chunk_spans(padded.height, seg);
    let col_spans = chunk_spans(padded.width, seg);
    let interior = |s: &Range<usize>| s.len() - 2 * RADIUS;
    let work: usize = col_spans.iter().map(interior).sum::<usize>()
        * row_spans.iter().map(interior).sum::<usize>();
    progress.start(2 * work);
    let _span = info_span!("hybrid_parity", chunks = row_spans.len() * col_spans.len()).entered();

    let mut sharp = Plane::zeros(padded.width, padded.height);
    let mut cleaned = Plane::zeros(padded.width, padded.height);
    for cs in &col_spans {
        for rs in &row_spans {
            let _chunk = debug_span!("chunk", rows = ?rs, cols = ?cs).entered();
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
    Ok(Nets::Parity { sharp, cleaned })
}

/// Hybrid output from the R-parity planes. R's fill_matrix + trimming maps original pixel
/// (r, c) to padded (r + 4, c + 4).
fn hybrid_parity_blend(plane: &Plane, sharp: &Plane, cleaned: &Plane, a: f64) -> Plane {
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
    out
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
