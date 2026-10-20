//! The models that use the star mask: Dual PSF (`DualPSF.R`) and Star Mask (`StarMask.R`).
//!
//! Dual PSF runs two PSF models on the image, `d` (the "PSF for DSO") and `s` (the "PSF
//! for Stars"), and mixes them through the star mask `m` as `s * m + d * (1 - m)` in the
//! region the models reach; the result is blended with the input by the aggressiveness
//! and clamped to [0, 1]. The R app builds the mask separately for each processing chunk;
//! [`Mode::Parity`] reproduces that, while [`Mode::Seamless`] uses one mask for the whole
//! image, the one Star Mask outputs.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;
use tracing::info_span;

use crate::features::RADIUS;
use crate::pipeline::{
    Keep, Mode, ModelKind, Nets, Params, Progress, channel, chunk_spans, eval_windows,
    hybrid_blend, kept, with_lightness,
};
use crate::starmask::{self, StarResponse, response_work};
use crate::{Error, Image, Plane, color, weights};

/// Dual PSF or Star Mask over `rows` x `cols` of `img`, reusing `cached` network outputs
/// and the whole image's star `response` if given. Returns the output, the network outputs
/// (if computed and `keep.nets`) and the response (if computed and `keep.response`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    img: &Image,
    rows: Range<usize>,
    cols: Range<usize>,
    params: &Params,
    cached: Option<&Nets>,
    response: Option<&StarResponse>,
    keep: Keep,
    progress: &Progress,
) -> Result<(Image, Option<Nets>, Option<StarResponse>), Error> {
    let (w, h) = (img.width, img.height);
    let dual = params.model == ModelKind::DualPsf;
    let chunked = dual && params.mode == Mode::Parity;
    // The pixels the networks reach: the region without the image's 4/5 px border.
    let kr = intersect(&rows, &kept(h));
    let kc = intersect(&cols, &kept(w));
    let reach = !kr.is_empty() && !kc.is_empty();

    let compute_response = !chunked && response.is_none();
    let chunks = if chunked && reach {
        parity_chunks(h, w, params.chunk_size, &kr, &kc)
    } else {
        Vec::new()
    };
    let compute_nets = dual && cached.is_none();
    let same_psf = weights::psf_index(params.psf_fwhm) == weights::psf_index(params.star_psf_fwhm);
    let mut total = 0;
    if compute_response {
        total += response_work(w * h);
    }
    total += chunks
        .iter()
        .map(|c| response_work(c.rows.len() * c.cols.len()))
        .sum::<usize>();
    if compute_nets {
        total += kr.len() * kc.len() * if same_psf { 1 } else { 2 };
    }
    progress.start(total.max(1));
    if total == 0 {
        progress.advance(1);
    }

    // The whole channel, when the mask needs it or Dual PSF processes the whole image.
    let full = (rows.len(), cols.len()) == (h, w);
    let whole = (compute_response || chunked || (dual && full))
        .then(|| info_span!("to_luv", width = w, height = h).in_scope(|| channel(img, 0..h, 0..w)));
    let fresh_response = if compute_response {
        let d = starmask::response(whole.as_ref().unwrap(), progress)?;
        Some(StarResponse::new(img, d))
    } else {
        None
    };
    let d = (!chunked).then(|| &fresh_response.as_ref().or(response).unwrap().d);
    let keep_response = |r: Option<StarResponse>| r.filter(|_| keep.response);

    if !dual {
        drop(whole);
        let m = starmask::mask(d.unwrap(), &params.mask, rows.clone(), cols.clone());
        let out = if img.is_color() && params.mode == Mode::Parity {
            // The app writes the mask into the lightness of an otherwise black (u = v = 0)
            // Luv image, giving grey RGB.
            let data = m
                .data
                .par_iter()
                .flat_map_iter(|&v| color::luv_to_srgb([v * 100.0, 0.0, 0.0]))
                .collect();
            Image::new(m.width, m.height, 3, data)
        } else {
            Image::new(m.width, m.height, 1, m.data)
        };
        let nets = (cached.is_none() && keep.nets).then_some(Nets::Empty);
        return Ok((out, nets, keep_response(fresh_response)));
    }

    let m = if !reach {
        Plane::zeros(0, 0)
    } else if chunked {
        chunk_masks(whole.as_ref().unwrap(), &chunks, params, &kr, &kc, progress)?
    } else {
        starmask::mask(d.unwrap(), &params.mask, kr.clone(), kc.clone())
    };
    // The response is no longer needed (before the networks add to the memory in use);
    // free it unless it is kept.
    let fresh_response = keep_response(fresh_response);

    let fresh_nets = if compute_nets && reach {
        // The networks read 9x9 windows: the kept region plus RADIUS on each side.
        let (src, rr, cc) = match &whole {
            Some(p) => (Cow::Borrowed(p), kr.clone(), kc.clone()),
            None => (
                Cow::Owned(channel(
                    img,
                    kr.start - RADIUS..kr.end + RADIUS,
                    kc.start - RADIUS..kc.end + RADIUS,
                )),
                RADIUS..RADIUS + kr.len(),
                RADIUS..RADIUS + kc.len(),
            ),
        };
        let pass = |fwhm: f64, which: &str| {
            let mlp = weights::psf(fwhm)?;
            info_span!("psf_pass", which)
                .in_scope(|| eval_windows(mlp, &src, rr.clone(), cc.clone(), progress))
        };
        let dso = pass(params.psf_fwhm, "dso")?;
        let star = if same_psf {
            None
        } else {
            Some(pass(params.star_psf_fwhm, "stars")?)
        };
        Some(Nets::Dual { dso, star })
    } else if compute_nets {
        Some(Nets::Dual {
            dso: Vec::new(),
            star: None,
        })
    } else {
        None
    };
    let Some(Nets::Dual { dso, star }) = fresh_nets.as_ref().or(cached) else {
        unreachable!("Dual PSF outputs are reused only from Dual PSF");
    };

    let base = match &whole {
        Some(p) if full => Cow::Borrowed(p),
        Some(p) => Cow::Owned(p.sub(rows.clone(), cols.clone())),
        None => Cow::Owned(channel(img, rows.clone(), cols.clone())),
    };
    let a = params.aggressiveness;
    let out = info_span!("blend").in_scope(|| {
        let (cw, kw) = (cols.len(), kc.len());
        let mut out = Plane::new(
            cw,
            rows.len(),
            base.data.iter().map(|&x| hybrid_blend(a, x, x)).collect(),
        );
        if reach {
            let (r0, c0) = (kr.start - rows.start, kc.start - cols.start);
            out.data[r0 * cw..(r0 + kr.len()) * cw]
                .par_chunks_mut(cw)
                .enumerate()
                .for_each(|(i, orow)| {
                    let xrow = &base.data[(r0 + i) * cw..(r0 + i + 1) * cw];
                    for j in 0..kw {
                        let k = i * kw + j;
                        let dv = dso[k];
                        let sv = star.as_ref().map_or(dv, |s| s[k]);
                        let mv = m.data[k];
                        // DualPSF.R: betavals * starM + tempvals * (1 - starM), then
                        // testsamp2 * Cleanval + tif1 * (1 - Cleanval), clamped.
                        let v = sv * mv + dv * (1.0 - mv);
                        orow[c0 + j] = hybrid_blend(a, v, xrow[c0 + j]);
                    }
                });
        }
        out
    });
    // Free what the conversion back doesn't need.
    drop(base);
    drop((whole, m));
    let fresh_nets = fresh_nets.filter(|_| keep.nets);
    let out = info_span!("from_luv").in_scope(|| with_lightness(img, rows, cols, out, params.mode));
    Ok((out, fresh_nets, fresh_response))
}

fn intersect(a: &Range<usize>, b: &Range<usize>) -> Range<usize> {
    let start = a.start.max(b.start);
    start..a.end.min(b.end).max(start)
}

/// A chunk of R's Dual PSF loop whose mask some pixels of the region take.
#[derive(Clone, Debug)]
struct Chunk {
    /// The chunk, in the zero-padded image.
    rows: Range<usize>,
    cols: Range<usize>,
    /// The pixels of the region whose mask comes from this chunk, in the padded image.
    own_rows: Range<usize>,
    own_cols: Range<usize>,
}

/// The chunks of the zero-padded `h` x `w` image (`chunk_spans`, as in `DualPSF.R`) whose
/// masks the network region `kr` x `kc` takes. R writes each chunk's interior in turn,
/// column blocks outer and row blocks inner, so a pixel keeps the value of the last chunk
/// along each axis whose interior contains it.
fn parity_chunks(
    h: usize,
    w: usize,
    seg: usize,
    kr: &Range<usize>,
    kc: &Range<usize>,
) -> Vec<Chunk> {
    // Owned parts along one axis, for the padded positions `want`.
    let owned = |n: usize, want: Range<usize>| -> Vec<(Range<usize>, Range<usize>)> {
        let spans = chunk_spans(n, seg);
        let mut owner = vec![usize::MAX; n];
        for (i, s) in spans.iter().enumerate() {
            owner[s.start + RADIUS..s.end - RADIUS].fill(i);
        }
        let mut out: Vec<(Range<usize>, Range<usize>)> = Vec::new();
        for p in want {
            let i = owner[p];
            assert_ne!(i, usize::MAX, "every kept pixel lies inside a chunk");
            match out.last_mut() {
                Some((span, own)) if *span == spans[i] => own.end = p + 1,
                _ => out.push((spans[i].clone(), p..p + 1)),
            }
        }
        out
    };
    let rows = owned(h + 2 * RADIUS, kr.start + RADIUS..kr.end + RADIUS);
    let cols = owned(w + 2 * RADIUS, kc.start + RADIUS..kc.end + RADIUS);
    let mut chunks = Vec::with_capacity(rows.len() * cols.len());
    for (cs, own_cols) in &cols {
        for (rs, own_rows) in &rows {
            chunks.push(Chunk {
                rows: rs.clone(),
                cols: cs.clone(),
                own_rows: own_rows.clone(),
                own_cols: own_cols.clone(),
            });
        }
    }
    chunks
}

/// The mask over `kr` x `kc` built chunk by chunk, as the R app does: each chunk's mask
/// comes from that chunk of the zero-padded channel.
fn chunk_masks(
    channel: &Plane,
    chunks: &[Chunk],
    params: &Params,
    kr: &Range<usize>,
    kc: &Range<usize>,
    progress: &Progress,
) -> Result<Plane, Error> {
    let _span = info_span!("dual_parity_masks", chunks = chunks.len()).entered();
    let padded = channel.padded(RADIUS);
    let masks = chunks
        .par_iter()
        .map(|c| {
            let t = padded.sub(c.rows.clone(), c.cols.clone());
            let d = starmask::response(&t, progress)?;
            let own_r = c.own_rows.start - c.rows.start..c.own_rows.end - c.rows.start;
            let own_c = c.own_cols.start - c.cols.start..c.own_cols.end - c.cols.start;
            Ok(starmask::mask(&d, &params.mask, own_r, own_c))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    drop(padded);
    let mut m = Plane::zeros(kc.len(), kr.len());
    for (c, cm) in chunks.iter().zip(&masks) {
        // Padded (r + 4, c + 4) is pixel (r, c); the mask plane starts at (kr.start, kc.start).
        let r0 = c.own_rows.start - RADIUS - kr.start;
        let c0 = c.own_cols.start - RADIUS - kc.start;
        m.write_region(r0..r0 + cm.height, c0..c0 + cm.width, &cm.data);
    }
    Ok(m)
}
