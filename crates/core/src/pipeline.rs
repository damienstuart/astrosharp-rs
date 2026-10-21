//! The AstroSharp processing pipeline, following `First3BetasFun.R` (PSF and
//! AstroClean models) and `Combined_Sharpen_and_Clean.R` (Hybrid model). The models that
//! use the star mask (Dual PSF and Star Mask) are in [`crate::dual`].

use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use rayon::prelude::*;
use tracing::{debug_span, info_span};

use crate::features::{self, N_FEATURES, RADIUS};
use crate::model::{Mlp, Scratch};
use crate::starmask::{MaskParams, StarResponse};
use crate::{Error, Image, Plane, color, dual, weights};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    /// PSF sharpening model (the app's "PSF Model (Pre-Beta)").
    Psf,
    /// AstroClean denoising model.
    AstroClean,
    /// PSF sharpening followed by AstroClean ("Hybrid Model").
    Hybrid,
    /// Two PSF models, one for the stars and one for everything else (the deep-sky object),
    /// mixed through the star mask ("Dual PSF").
    DualPsf,
    /// The star mask itself ("Star Mask").
    StarMask,
}

impl ModelKind {
    pub const ALL: [ModelKind; 5] = [
        ModelKind::Psf,
        ModelKind::AstroClean,
        ModelKind::Hybrid,
        ModelKind::DualPsf,
        ModelKind::StarMask,
    ];

    /// Short name, e.g. `Dual PSF`.
    pub fn name(self) -> &'static str {
        match self {
            ModelKind::Psf => "PSF",
            ModelKind::AstroClean => "AstroClean",
            ModelKind::Hybrid => "Hybrid",
            ModelKind::DualPsf => "Dual PSF",
            ModelKind::StarMask => "Star Mask",
        }
    }

    /// Uses [`Params::psf_fwhm`].
    pub fn uses_psf(self) -> bool {
        matches!(
            self,
            ModelKind::Psf | ModelKind::Hybrid | ModelKind::DualPsf
        )
    }

    /// Uses [`Params::mask`].
    pub fn uses_star_mask(self) -> bool {
        matches!(self, ModelKind::DualPsf | ModelKind::StarMask)
    }

    /// Uses [`Params::aggressiveness`].
    pub fn uses_aggressiveness(self) -> bool {
        self != ModelKind::StarMask
    }

    /// Processes in chunks in [`Mode::Parity`], so [`Params::chunk_size`] matters.
    pub fn chunked_in_parity(self) -> bool {
        matches!(self, ModelKind::Hybrid | ModelKind::DualPsf)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Whole-image passes at full precision: Hybrid avoids the faint seams the R app's
    /// chunked processing leaves, Dual PSF uses one star mask for the whole image (the one
    /// Star Mask shows), and colour output is not rounded to 5 decimals. Grey PSF,
    /// AstroClean and Star Mask output is identical to `Parity`.
    #[default]
    Seamless,
    /// Reproduces the R app's output exactly: chunked Hybrid, Dual PSF with a star mask per
    /// chunk, colour rounded to 5 decimals, and Star Mask of a colour image as grey RGB.
    Parity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub model: ModelKind,
    /// Selects the PSF model: 1–8 pixels in steps of 0.25. For Dual PSF, the model used
    /// away from stars (the app's "PSF for DSO").
    pub psf_fwhm: f64,
    /// Dual PSF: the PSF model used on stars ("PSF for Stars").
    pub star_psf_fwhm: f64,
    /// Blend between the model output (1) and the input (0).
    pub aggressiveness: f64,
    pub mode: Mode,
    /// Chunk size for Hybrid and Dual PSF in `Mode::Parity` (the app's "Processing Chunk
    /// Size").
    pub chunk_size: usize,
    /// Star mask settings (Dual PSF and Star Mask).
    pub mask: MaskParams,
}

impl Default for Params {
    /// The R app's defaults (except the aggressiveness, which the app's later versions
    /// raised to 1).
    fn default() -> Self {
        Params {
            model: ModelKind::Psf,
            psf_fwhm: 3.0,
            star_psf_fwhm: 3.0,
            aggressiveness: 0.66,
            mode: Mode::Seamless,
            chunk_size: 325,
            mask: MaskParams::default(),
        }
    }
}

impl Params {
    /// Short description, e.g. `Hybrid (PSF 3, aggressiveness 0.66, seamless)`.
    pub fn describe(&self) -> String {
        let m = self.model;
        let mut parts = Vec::new();
        if m.uses_psf() {
            parts.push(format!("PSF {}", self.psf_fwhm));
        }
        if m == ModelKind::DualPsf {
            parts.push(format!("star PSF {}", self.star_psf_fwhm));
        }
        if m.uses_aggressiveness() {
            parts.push(format!("aggressiveness {:.2}", self.aggressiveness));
        }
        if m.uses_star_mask() {
            parts.push(self.mask.describe());
        }
        if matches!(
            m,
            ModelKind::Hybrid | ModelKind::DualPsf | ModelKind::StarMask
        ) {
            parts.push(match self.mode {
                Mode::Seamless => "seamless".to_string(),
                Mode::Parity if m.chunked_in_parity() => {
                    format!("R parity, chunk size {}", self.chunk_size)
                }
                Mode::Parity => "R parity".to_string(),
            });
        }
        format!("{} ({})", m.name(), parts.join(", "))
    }

    /// Checks the settings the model uses.
    pub fn validate(&self) -> Result<(), Error> {
        let a = self.aggressiveness;
        if !(0.0..=1.0).contains(&a) {
            return Err(Error::InvalidParams(format!(
                "aggressiveness must be between 0 and 1, got {a}"
            )));
        }
        if self.model.uses_psf() {
            weights::psf(self.psf_fwhm)?;
        }
        if self.model == ModelKind::DualPsf {
            weights::psf(self.star_psf_fwhm)?;
        }
        if self.model.uses_star_mask() {
            self.mask.validate()?;
        }
        if self.model.chunked_in_parity()
            && self.mode == Mode::Parity
            && self.chunk_size < MIN_CHUNK_SIZE
        {
            return Err(Error::InvalidParams(format!(
                "chunk size must be at least {MIN_CHUNK_SIZE}"
            )));
        }
        Ok(())
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

    pub(crate) fn start(&self, total: usize) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total as u64, Ordering::Relaxed);
    }

    pub(crate) fn advance(&self, n: usize) {
        self.done.fetch_add(n as u64, Ordering::Relaxed);
    }
}

/// The settings that change the neural-network outputs: everything except the
/// aggressiveness and the star mask settings; for the single models and Dual PSF also
/// the mode (which there only affects colour rounding and the mask).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetKey {
    model: ModelKind,
    /// PSF model index (see [`weights::psf_index`]); `None` if the model uses none.
    psf: Option<usize>,
    /// Dual PSF only: the star PSF model index.
    star_psf: Option<usize>,
    /// Hybrid only.
    mode: Option<Mode>,
    /// Hybrid in parity mode only.
    chunk_size: Option<usize>,
}

impl NetKey {
    pub fn of(params: &Params) -> NetKey {
        let m = params.model;
        let hybrid = m == ModelKind::Hybrid;
        NetKey {
            model: m,
            psf: m
                .uses_psf()
                .then(|| weights::psf_index(params.psf_fwhm))
                .flatten(),
            star_psf: (m == ModelKind::DualPsf)
                .then(|| weights::psf_index(params.star_psf_fwhm))
                .flatten(),
            mode: hybrid.then_some(params.mode),
            chunk_size: (hybrid && params.mode == Mode::Parity).then_some(params.chunk_size),
        }
    }
}

/// The neural-network outputs for one image (or region of it): everything [`process`]
/// computes except the final blend, which depends only on the aggressiveness (and for
/// Dual PSF the star mask). See [`process_with`].
#[derive(Clone, Debug)]
pub struct NetOutputs {
    key: NetKey,
    width: usize,
    height: usize,
    channels: usize,
    /// The region of the image they were computed for.
    rows: Range<usize>,
    cols: Range<usize>,
    nets: Nets,
}

impl NetOutputs {
    /// Whether these outputs fit `img` (by size and channels) and give the same network
    /// outputs as `params`.
    pub fn matches(&self, img: &Image, params: &Params) -> bool {
        self.matches_region(img, 0..img.height, 0..img.width, params)
    }

    /// Like [`matches`](Self::matches), for outputs of [`process_region`].
    pub fn matches_region(
        &self,
        img: &Image,
        rows: Range<usize>,
        cols: Range<usize>,
        params: &Params,
    ) -> bool {
        self.key == NetKey::of(params)
            && (self.width, self.height, self.channels) == (img.width, img.height, img.channels)
            && (&self.rows, &self.cols) == (&rows, &cols)
    }

    /// The settings these outputs were computed with.
    pub fn key(&self) -> NetKey {
        self.key
    }

    /// Memory these outputs take (see [`net_outputs_bytes`]).
    pub fn bytes(&self) -> usize {
        let values = match &self.nets {
            Nets::Single(nn) => nn.len(),
            Nets::Seamless { sharp, cleaned } => sharp.data.len() + cleaned.len(),
            Nets::Parity { sharp, cleaned } => sharp.data.len() + cleaned.data.len(),
            Nets::Dual { dso, star } => dso.len() + star.as_ref().map_or(0, Vec::len),
            Nets::Empty => 0,
        };
        values * size_of::<f64>()
    }
}

/// Network outputs for one channel, in the layout each model's blend reads.
#[derive(Clone, Debug)]
pub(crate) enum Nets {
    /// PSF or AstroClean output over the kept region, row-major.
    Single(Vec<f64>),
    /// Hybrid, seamless: the sharpened channel and the cleaned kept region (row-major).
    Seamless { sharp: Plane, cleaned: Vec<f64> },
    /// Hybrid, R parity: sharpened and cleaned padded planes.
    Parity { sharp: Plane, cleaned: Plane },
    /// Dual PSF: both PSF models over the kept part of the region, row-major; `star` is
    /// `None` when both use the same model.
    Dual {
        dso: Vec<f64>,
        star: Option<Vec<f64>>,
    },
    /// Star Mask, which runs no networks.
    Empty,
}

/// Earlier results [`process_with`] and [`process_region`] can reuse; each is used only if
/// it fits the image and settings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Reuse<'a> {
    pub nets: Option<&'a NetOutputs>,
    /// The star response of the whole image (Dual PSF in seamless mode, and Star Mask).
    pub response: Option<&'a StarResponse>,
}

/// Which newly computed results [`process_with`] and [`process_region`] return.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Keep {
    pub nets: bool,
    pub response: bool,
}

impl Keep {
    pub const NONE: Keep = Keep {
        nets: false,
        response: false,
    };
    pub const ALL: Keep = Keep {
        nets: true,
        response: true,
    };
}

/// Results computed by [`process_with`] or [`process_region`] that were asked for with
/// [`Keep`]; `None` where they were reused (or not needed).
#[derive(Debug, Default)]
pub struct Fresh {
    pub nets: Option<NetOutputs>,
    pub response: Option<StarResponse>,
}

/// Processes a grey or RGB image. Colour images are converted to CIE Luv and only the
/// lightness channel is processed, as in the R app.
pub fn process(img: &Image, params: &Params, progress: &Progress) -> Result<Image, Error> {
    process_with(img, params, Reuse::default(), Keep::NONE, progress).map(|(out, _)| out)
}

/// Like [`process`], but reuses earlier results that fit the image and settings, so for
/// example only the blend runs when just the aggressiveness changed. Returns the results
/// it computed that `keep` asks for; network outputs take [`net_outputs_bytes`] of memory
/// and the star response [`crate::response_bytes`].
pub fn process_with(
    img: &Image,
    params: &Params,
    reuse: Reuse,
    keep: Keep,
    progress: &Progress,
) -> Result<(Image, Fresh), Error> {
    process_region(
        img,
        0..img.height,
        0..img.width,
        params,
        reuse,
        keep,
        progress,
    )
}

/// Extra context processed around a region by the models that only look at nearby
/// pixels (two stacked 9x9 windows for Hybrid).
pub const REGION_MARGIN: usize = 8;

/// Processes `rows` x `cols` of `img`, giving exactly what [`process`] gives there (except
/// Hybrid in parity mode, whose chunks depend on the region). The star mask, which
/// depends on the whole image, comes from the whole image (or a reused response).
pub fn process_region(
    img: &Image,
    rows: Range<usize>,
    cols: Range<usize>,
    params: &Params,
    reuse: Reuse,
    keep: Keep,
    progress: &Progress,
) -> Result<(Image, Fresh), Error> {
    let span = process_span(img, params, &rows, &cols);
    let _enter = span.enter();
    if img.width < MIN_SIZE || img.height < MIN_SIZE {
        return Err(Error::InvalidParams(format!(
            "image must be at least {MIN_SIZE}x{MIN_SIZE} pixels"
        )));
    }
    if rows.is_empty() || cols.is_empty() || rows.end > img.height || cols.end > img.width {
        return Err(Error::InvalidParams(format!(
            "region {rows:?} x {cols:?} is not inside the {}x{} image",
            img.width, img.height
        )));
    }
    params.validate()?;
    let cached = reuse
        .nets
        .filter(|n| n.matches_region(img, rows.clone(), cols.clone(), params))
        .map(|n| &n.nets);
    span.record("reused", cached.is_some());

    let (out, nets, response) = if params.model.uses_star_mask() {
        let response = reuse.response.filter(|r| r.matches(img));
        dual::run(
            img,
            rows.clone(),
            cols.clone(),
            params,
            cached,
            response,
            keep,
            progress,
        )?
    } else if (rows.len(), cols.len()) == (img.height, img.width) {
        let (out, nets) = run(img, params, cached, keep.nets, progress)?;
        (out, nets, None)
    } else {
        // Process the region with enough context that every result in it is exact.
        let r = with_margin(&rows, img.height);
        let c = with_margin(&cols, img.width);
        let crop = img.crop(r.clone(), c.clone());
        let (out, nets) = run(&crop, params, cached, keep.nets, progress)?;
        let out = out.crop(
            rows.start - r.start..rows.end - r.start,
            cols.start - c.start..cols.end - c.start,
        );
        (out, nets, None)
    };
    let nets = nets.filter(|_| keep.nets).map(|nets| NetOutputs {
        key: NetKey::of(params),
        width: img.width,
        height: img.height,
        channels: img.channels,
        rows,
        cols,
        nets,
    });
    Ok((out, Fresh { nets, response }))
}

/// `r` widened by [`REGION_MARGIN`] on each side, within `0..n`, and to at least
/// [`MIN_SIZE`] pixels.
fn with_margin(r: &Range<usize>, n: usize) -> Range<usize> {
    let start = r.start.saturating_sub(REGION_MARGIN);
    let end = (r.end + REGION_MARGIN).max(start + MIN_SIZE).min(n);
    end.saturating_sub(MIN_SIZE).min(start)..end
}

/// Memory taken by the network outputs [`process_with`] returns for a `width` x `height`
/// image (for colour images only the lightness is processed).
pub fn net_outputs_bytes(width: usize, height: usize, params: &Params) -> usize {
    let (kw, kh) = (
        width.saturating_sub(RADIUS + 5),
        height.saturating_sub(RADIUS + 5),
    );
    let values = match (params.model, params.mode) {
        (ModelKind::Psf | ModelKind::AstroClean, _) => kw * kh,
        (ModelKind::Hybrid, Mode::Seamless) => width * height + kw * kh,
        (ModelKind::Hybrid, Mode::Parity) => 2 * (width + 2 * RADIUS) * (height + 2 * RADIUS),
        (ModelKind::DualPsf, _) => {
            let same =
                weights::psf_index(params.psf_fwhm) == weights::psf_index(params.star_psf_fwhm);
            kw * kh * if same { 1 } else { 2 }
        }
        (ModelKind::StarMask, _) => 0,
    };
    values * size_of::<f64>()
}

/// The `process` span: image size, region and settings.
fn process_span(
    img: &Image,
    params: &Params,
    rows: &Range<usize>,
    cols: &Range<usize>,
) -> tracing::Span {
    let m = params.model;
    let whole = (rows.len(), cols.len()) == (img.height, img.width);
    info_span!(
        "process",
        width = img.width,
        height = img.height,
        color = img.is_color(),
        region = (!whole).then(|| format!(
            "{}x{} at {},{}",
            cols.len(),
            rows.len(),
            cols.start,
            rows.start
        )),
        model = ?m,
        psf_fwhm = m.uses_psf().then_some(params.psf_fwhm),
        star_psf_fwhm = (m == ModelKind::DualPsf).then_some(params.star_psf_fwhm),
        aggressiveness = m.uses_aggressiveness().then_some(params.aggressiveness),
        mask = m.uses_star_mask().then(|| params.mask.describe()),
        mode = ?params.mode,
        chunk_size = (m.chunked_in_parity() && params.mode == Mode::Parity)
            .then_some(params.chunk_size),
        reused = tracing::field::Empty,
    )
}

/// PSF, AstroClean or Hybrid on a whole image, reusing `cached` network outputs if given.
/// With `keep`, newly computed outputs are returned; without it they are freed before the
/// colour conversion.
fn run(
    img: &Image,
    params: &Params,
    cached: Option<&Nets>,
    keep: bool,
    progress: &Progress,
) -> Result<(Image, Option<Nets>), Error> {
    let (w, h) = (img.width, img.height);
    if !img.is_color() {
        let plane = Plane::new(w, h, img.data.clone());
        let (out, nets) = plane_stage(&plane, params, cached, progress)?;
        return Ok((Image::new(w, h, 1, out.data), nets.filter(|_| keep)));
    }
    // Only the lightness is processed. Its u and v are recomputed from the source pixels at
    // the end rather than kept: a full Luv copy would be as large as the image.
    let lightness = info_span!("to_luv").in_scope(|| channel(img, 0..h, 0..w));
    let (out, nets) = plane_stage(&lightness, params, cached, progress)?;
    drop(lightness);
    // Without `keep`, the network outputs are freed here, before the conversion back.
    let nets = nets.filter(|_| keep);
    let out = info_span!("from_luv").in_scope(|| with_lightness(img, 0..h, 0..w, out, params.mode));
    Ok((out, nets))
}

/// The channel the models process over `rows` x `cols` of `img`: the grey values, or the
/// CIE Luv lightness / 100 of colour images.
pub(crate) fn channel(img: &Image, rows: Range<usize>, cols: Range<usize>) -> Plane {
    let (cw, ch) = (cols.len(), rows.len());
    let mut data = vec![0.0; cw * ch];
    let n = img.channels;
    data.par_chunks_mut(cw).enumerate().for_each(|(i, out)| {
        let start = ((rows.start + i) * img.width + cols.start) * n;
        let src = &img.data[start..start + cw * n];
        if n == 1 {
            out.copy_from_slice(src);
        } else {
            for (o, p) in out.iter_mut().zip(src.as_chunks::<3>().0) {
                *o = color::srgb_to_luv(*p)[0] / 100.0;
            }
        }
    });
    Plane::new(cw, ch, data)
}

/// `rows` x `cols` of `img` with its processed channel replaced by `l` (see [`channel`]):
/// for colour images, the lightness goes back with each pixel's own u and v.
pub(crate) fn with_lightness(
    img: &Image,
    rows: Range<usize>,
    cols: Range<usize>,
    l: Plane,
    mode: Mode,
) -> Image {
    let (cw, ch) = (cols.len(), rows.len());
    assert_eq!((l.width, l.height), (cw, ch));
    if !img.is_color() {
        return Image::new(cw, ch, 1, l.data);
    }
    // Written in place: collecting a non-indexed iterator would hold the output twice.
    let mut data = vec![0.0; cw * ch * 3];
    data.par_chunks_mut(cw * 3)
        .zip(l.data.par_chunks(cw))
        .enumerate()
        .for_each(|(i, (out, lrow))| {
            let start = ((rows.start + i) * img.width + cols.start) * 3;
            let src = &img.data[start..start + cw * 3];
            for ((rgb, p), &lv) in out
                .as_chunks_mut::<3>()
                .0
                .iter_mut()
                .zip(src.as_chunks::<3>().0)
                .zip(lrow)
            {
                let [_, u, v] = color::srgb_to_luv(*p);
                let luv = [lv * 100.0, u, v];
                // R rounds to 5 decimals; only parity mode reproduces that.
                *rgb = match mode {
                    Mode::Parity => color::luv_to_srgb(luv),
                    Mode::Seamless => color::luv_to_srgb_unrounded(luv),
                };
            }
        });
    Image::new(cw, ch, 3, data)
}

/// Processes one channel (grey values, or Luv lightness / 100).
pub fn process_plane(plane: &Plane, params: &Params, progress: &Progress) -> Result<Plane, Error> {
    let img = Image::new(plane.width, plane.height, 1, plane.data.clone());
    process(&img, params, progress).map(|out| Plane::new(out.width, out.height, out.data))
}

/// Runs the networks unless `cached` has their outputs, and blends. Returns the network
/// outputs it computed, if any.
fn plane_stage(
    plane: &Plane,
    params: &Params,
    cached: Option<&Nets>,
    progress: &Progress,
) -> Result<(Plane, Option<Nets>), Error> {
    let a = params.aggressiveness;
    match cached {
        Some(cached) => {
            // Nothing left to wait for: show the work as done.
            progress.start(1);
            progress.advance(1);
            let out = info_span!("blend").in_scope(|| blend_plane(plane, cached, a));
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
                Mode::Parity => hybrid_parity_nets(psf, clean, plane, params.chunk_size, progress),
            }
        }
        ModelKind::DualPsf | ModelKind::StarMask => {
            unreachable!("star mask models run in crate::dual")
        }
    }
}

/// Combines the network outputs with the input `plane` for aggressiveness `a`.
fn blend_plane(plane: &Plane, nets: &Nets, a: f64) -> Plane {
    match nets {
        Nets::Single(nn) => single_blend(plane, nn, a),
        Nets::Seamless { sharp, cleaned } => hybrid_seamless_blend(plane, sharp, cleaned, a),
        Nets::Parity { sharp, cleaned } => hybrid_parity_blend(plane, sharp, cleaned, a),
        Nets::Dual { .. } | Nets::Empty => unreachable!("star mask models blend in crate::dual"),
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
pub(crate) fn kept(n: usize) -> Range<usize> {
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
pub(crate) fn hybrid_blend(a: f64, clean: f64, sharp: f64) -> f64 {
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
    drop(padded);
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
pub(crate) fn chunk_spans(n: usize, seg: usize) -> Vec<Range<usize>> {
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
pub(crate) fn eval_windows(
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
