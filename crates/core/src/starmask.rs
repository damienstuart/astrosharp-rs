//! The R app's star mask (`starmaskfun` in `DualPSF.R`, and `StarMask.R`):
//!
//! 1. `D` = the maximum over scales s = 2, 4, ..., 20 of `s^2 * det(Hessian(isoblur(T, s)))`,
//!    which peaks on star-like blobs (the [`response`], which depends only on the image);
//! 2. `B0 = D > q`, where `q` is the `sensitivity` percentile of a 1000-pixel sample of `D`;
//! 3. CImg's `label(B0)` numbers the 4-connected regions of both values in scan order
//!    (down each column, then the next column), and `B = label > fix_mask`;
//! 4. the mask is `pbeta(isoblur(B, feathering), blend, blend2)`.

use std::ops::Range;

use rayon::prelude::*;
use tracing::info_span;

use crate::pbeta::PBeta;
use crate::pipeline::Progress;
use crate::{Error, Image, Plane, cimg};

/// Blur scales of the multi-scale response: R's `seq(2, 20, l = 10)`.
pub const SCALES: [f64; 10] = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0];

/// Pixels per unit of progress for one response scale (a scale costs about this much less
/// than a network evaluation of one pixel).
const RESPONSE_PIXELS_PER_UNIT: usize = 32;

/// Settings of the star mask, with the R app's names in quotes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskParams {
    /// "Star Sensitivity": percentile of the response above which pixels are stars.
    pub sensitivity: f64,
    /// "Fix Mask": regions numbered up to this (in scan order) are left out of the mask;
    /// region 0 is normally the background.
    pub fix_mask: f64,
    /// "Feathering": blur sigma of the mask edges (below 0.1, no blur).
    pub feathering: f64,
    /// "Blend" and "Blend 2": shape parameters of the beta CDF that maps the feathered
    /// mask to [0, 1].
    pub blend: f64,
    pub blend2: f64,
}

impl Default for MaskParams {
    /// The R app's defaults.
    fn default() -> Self {
        MaskParams {
            sensitivity: 98.0,
            fix_mask: 2.0,
            feathering: 5.0,
            blend: 9.0,
            blend2: 100.0,
        }
    }
}

impl MaskParams {
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |what: &str, v: f64| {
            Err(Error::InvalidParams(format!(
                "star mask {what} is out of range: {v}"
            )))
        };
        if !(0.0..=100.0).contains(&self.sensitivity) {
            return bad("sensitivity (0-100)", self.sensitivity);
        }
        if !self.fix_mask.is_finite() {
            return bad("fix mask", self.fix_mask);
        }
        if !(self.feathering >= 0.0 && self.feathering.is_finite()) {
            return bad("feathering (>= 0)", self.feathering);
        }
        for (what, v) in [("blend (> 0)", self.blend), ("blend 2 (> 0)", self.blend2)] {
            if !(v > 0.0 && v.is_finite()) {
                return bad(what, v);
            }
        }
        Ok(())
    }

    /// Short description, e.g. `mask 98/2/5/9/100` (sensitivity, fix mask, feathering,
    /// blend, blend 2).
    pub fn describe(&self) -> String {
        format!(
            "mask {}/{}/{}/{}/{}",
            self.sensitivity, self.fix_mask, self.feathering, self.blend, self.blend2
        )
    }
}

/// The multi-scale star response of a whole image (of its lightness, for colour images).
/// It depends only on the image, so it can be kept while the mask settings change.
#[derive(Clone, Debug)]
pub struct StarResponse {
    width: usize,
    height: usize,
    channels: usize,
    pub(crate) d: Plane,
}

impl StarResponse {
    pub(crate) fn new(img: &Image, d: Plane) -> StarResponse {
        assert_eq!((d.width, d.height), (img.width, img.height));
        StarResponse {
            width: img.width,
            height: img.height,
            channels: img.channels,
            d,
        }
    }

    /// Whether this is the response of an image of `img`'s size and type.
    pub fn matches(&self, img: &Image) -> bool {
        (self.width, self.height, self.channels) == (img.width, img.height, img.channels)
    }

    /// Memory it takes (see [`response_bytes`]).
    pub fn bytes(&self) -> usize {
        self.d.data.len() * size_of::<f64>()
    }
}

/// Memory a [`StarResponse`] of a `width` x `height` image takes.
pub fn response_bytes(width: usize, height: usize) -> usize {
    width * height * size_of::<f64>()
}

/// Progress units [`response`] reports for a plane of `pixels` pixels.
pub(crate) fn response_work(pixels: usize) -> usize {
    SCALES.len() * pixels.div_ceil(RESPONSE_PIXELS_PER_UNIT)
}

/// The multi-scale response `D` of `t` (step 1 of the module docs).
pub fn response(t: &Plane, progress: &Progress) -> Result<Plane, Error> {
    let _span = info_span!("star_response", width = t.width, height = t.height).entered();
    let mut d = Plane::new(t.width, t.height, vec![f64::NEG_INFINITY; t.data.len()]);
    let mut g = t.clone();
    for (i, &s) in SCALES.iter().enumerate() {
        if progress.is_cancelled() {
            return Err(Error::Cancelled);
        }
        if i > 0 {
            g.data.copy_from_slice(&t.data);
        }
        cimg::isoblur(&mut g, s);
        cimg::fold_hessian_det(&g, s * s, &mut d);
        progress.advance(t.data.len().div_ceil(RESPONSE_PIXELS_PER_UNIT));
    }
    Ok(d)
}

/// imager's `threshold(d, "<sensitivity>%")`: the type-7 quantile of R's 1000-pixel sample
/// `d[round(seq(1, N, l = 1000))]` (column-major), or of all pixels if there are at most
/// 1000.
pub fn threshold(d: &Plane, sensitivity: f64) -> f64 {
    let (w, h) = (d.width, d.height);
    let n = w * h;
    // Column-major (R) index to the plane's row-major one.
    let at = |i: usize| d.data[(i % h) * w + i / h];
    let mut v: Vec<f64> = if n > 1000 {
        sample_indices(n).into_iter().map(at).collect()
    } else {
        (0..n).map(at).collect()
    };
    // The app builds the threshold as paste0(sensi, "%"): 15 significant digits.
    let qt: f64 = format!("{sensitivity:.14e}").parse().unwrap();
    quantile7(&mut v, qt / 100.0)
}

/// 0-based indices of R's `round(seq(1, n, l = 1000))`, for `n > 1000`.
pub fn sample_indices(n: usize) -> Vec<usize> {
    let by = (n as f64 - 1.0) / 999.0;
    (0..1000)
        .map(|k| {
            let v = match k {
                0 => 1.0,
                999 => n as f64,
                _ => 1.0 + k as f64 * by,
            };
            v.round_ties_even() as usize - 1
        })
        .collect()
}

/// R's `quantile(x, p, type = 7)` (sorts `x`).
pub fn quantile7(x: &mut [f64], p: f64) -> f64 {
    x.sort_by(f64::total_cmp);
    let n = x.len();
    let index = 1.0 + (n - 1) as f64 * p;
    let lo = index.floor();
    let hi = index.ceil();
    let qs = x[lo as usize - 1];
    let xhi = x[hi as usize - 1];
    if index > lo && xhi != qs {
        let h = index - lo;
        (1.0 - h) * qs + h * xhi
    } else {
        qs
    }
}

/// A maximal horizontal run of equal values in one row.
#[derive(Clone, Copy, Debug)]
struct Run {
    row: u32,
    start: u32,
    end: u32,
    value: bool,
}

/// The 4-connected regions of a boolean plane, numbered like CImg's `label()`.
struct Regions {
    runs: Vec<Run>,
    /// Label of each run's region.
    labels: Vec<u32>,
    count: usize,
}

impl Regions {
    /// Regions of `value(r, c)` over a `w` x `h` plane.
    fn find(w: usize, h: usize, value: impl Fn(usize, usize) -> bool + Sync) -> Regions {
        let rows: Vec<Vec<Run>> = (0..h)
            .into_par_iter()
            .map(|r| {
                let mut runs = Vec::new();
                let mut start = 0;
                for c in 1..=w {
                    if c == w || value(r, c) != value(r, start) {
                        runs.push(Run {
                            row: r as u32,
                            start: start as u32,
                            end: c as u32,
                            value: value(r, start),
                        });
                        start = c;
                    }
                }
                runs
            })
            .collect();
        let mut first = Vec::with_capacity(h + 1);
        let mut runs = Vec::new();
        for row in rows {
            first.push(runs.len());
            runs.extend(row);
        }
        first.push(runs.len());

        // Union runs that touch vertically and have the same value.
        let mut parent: Vec<u32> = (0..runs.len() as u32).collect();
        for r in 1..h {
            let (mut i, mut j) = (first[r - 1], first[r]);
            while i < first[r] && j < first[r + 1] {
                let (a, b) = (runs[i], runs[j]);
                if a.value == b.value && a.start < b.end && b.start < a.end {
                    union(&mut parent, i as u32, j as u32);
                }
                if a.end < b.end {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }

        // CImg numbers regions by their first pixel in column-major order; a run's first
        // pixel is its start, at column-major index start * h + row.
        let mut key = vec![u64::MAX; runs.len()];
        let roots: Vec<u32> = (0..runs.len() as u32)
            .map(|i| find(&mut parent, i))
            .collect();
        for (run, &root) in runs.iter().zip(&roots) {
            let k = run.start as u64 * h as u64 + run.row as u64;
            key[root as usize] = key[root as usize].min(k);
        }
        let mut order: Vec<u32> = (0..runs.len() as u32)
            .filter(|&i| roots[i as usize] == i)
            .collect();
        order.sort_by_key(|&i| key[i as usize]);
        let mut label_of_root = vec![0u32; runs.len()];
        for (label, &root) in order.iter().enumerate() {
            label_of_root[root as usize] = label as u32;
        }
        let labels = roots.iter().map(|&r| label_of_root[r as usize]).collect();
        Regions {
            runs,
            labels,
            count: order.len(),
        }
    }

    /// Writes `f(label)` into every pixel of each region.
    fn paint(&self, w: usize, h: usize, f: impl Fn(u32) -> f64) -> Plane {
        let mut out = Plane::zeros(w, h);
        for (run, &label) in self.runs.iter().zip(&self.labels) {
            let row = run.row as usize * w;
            out.data[row + run.start as usize..row + run.end as usize].fill(f(label));
        }
        out
    }
}

fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let p = parent[i as usize];
        parent[i as usize] = parent[p as usize];
        i = p;
    }
    i
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb) as usize] = ra.min(rb);
    }
}

/// CImg's `label()` of a boolean plane (4-connectivity, both values labelled), as a plane
/// of label numbers.
pub fn label(b: &Plane) -> Plane {
    let (w, h) = (b.width, b.height);
    let regions = Regions::find(w, h, |r, c| b.data[r * w + c] != 0.0);
    regions.paint(w, h, |l| l as f64)
}

/// `B = label(D > thr) > fix_mask`, as 0/1 values.
pub fn binary_mask(d: &Plane, thr: f64, fix_mask: f64) -> Plane {
    let (w, h) = (d.width, d.height);
    let regions = Regions::find(w, h, |r, c| d.data[r * w + c] > thr);
    tracing::Span::current().record("regions", regions.count);
    regions.paint(w, h, |l| if l as f64 > fix_mask { 1.0 } else { 0.0 })
}

/// The star mask from the response `d`, over `rows` x `cols` of it. Thresholding,
/// labelling and feathering use the whole response; only the final beta CDF is limited to
/// the region.
pub fn mask(d: &Plane, p: &MaskParams, rows: Range<usize>, cols: Range<usize>) -> Plane {
    let thr = threshold(d, p.sensitivity);
    let span = info_span!(
        "star_mask",
        width = d.width,
        height = d.height,
        threshold = thr,
        regions = tracing::field::Empty,
    );
    let _enter = span.enter();
    let mut b = binary_mask(d, thr, p.fix_mask);
    cimg::isoblur(&mut b, p.feathering);
    let mut out = if (rows.len(), cols.len()) == (d.height, d.width) {
        b
    } else {
        b.sub(rows, cols)
    };
    let pb = PBeta::new(p.blend, p.blend2);
    out.data
        .par_chunks_mut(4096)
        .for_each(|c| c.iter_mut().for_each(|v| *v = pb.at(*v)));
    out
}

/// The star mask of `t`, as the R app computes it for one image or chunk.
pub fn star_mask(t: &Plane, p: &MaskParams, progress: &Progress) -> Result<Plane, Error> {
    let d = response(t, progress)?;
    Ok(mask(&d, p, 0..t.height, 0..t.width))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bool_plane(rows: &[&str]) -> Plane {
        let h = rows.len();
        let w = rows[0].len();
        let data = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| if b == b'#' { 1.0 } else { 0.0 }))
            .collect();
        Plane::new(w, h, data)
    }

    /// Reference labelling: flood fill from each unlabelled pixel in column-major order.
    fn label_reference(b: &Plane) -> Vec<f64> {
        let (w, h) = (b.width, b.height);
        let mut lab = vec![f64::NAN; w * h];
        let mut next = 0.0;
        for c in 0..w {
            for r in 0..h {
                if !lab[r * w + c].is_nan() {
                    continue;
                }
                let v = b.at(r, c);
                let mut stack = vec![(r, c)];
                lab[r * w + c] = next;
                while let Some((r, c)) = stack.pop() {
                    let mut push = |r: usize, c: usize| {
                        if lab[r * w + c].is_nan() && b.at(r, c) == v {
                            lab[r * w + c] = next;
                            stack.push((r, c));
                        }
                    };
                    if r > 0 {
                        push(r - 1, c);
                    }
                    if r + 1 < h {
                        push(r + 1, c);
                    }
                    if c > 0 {
                        push(r, c - 1);
                    }
                    if c + 1 < w {
                        push(r, c + 1);
                    }
                }
                next += 1.0;
            }
        }
        lab
    }

    #[test]
    fn labels_follow_column_major_first_pixels() {
        let b = bool_plane(&[
            "....#....", //
            ".##.#..#.", //
            ".#.....#.", //
            "...###.#.", //
            "#..#.#...", //
            "...###..#", //
        ]);
        let ours = label(&b);
        assert_eq!(ours.data, label_reference(&b));
        // Region 0 is the background touching the top-left pixel; the first star found
        // scanning down the columns is the lone pixel at (4, 0).
        assert_eq!(ours.at(0, 0), 0.0);
        assert_eq!(ours.at(4, 0), 1.0);
        // The hole inside the ring is its own background region.
        assert_ne!(ours.at(4, 4), 0.0);
    }

    #[test]
    fn labels_match_flood_fill_on_noise() {
        let mut state = 12345u64;
        let data = (0..57 * 43)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (state >> 33).is_multiple_of(3) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let b = Plane::new(57, 43, data);
        assert_eq!(label(&b).data, label_reference(&b));
    }

    #[test]
    fn sample_indices_match_r() {
        // R: round(seq(1, 1001, l = 1000))[c(1:3, 500, 999, 1000)] - 1
        let idx = sample_indices(1001);
        assert_eq!(&idx[..3], &[0, 1, 2]);
        assert_eq!(idx[999], 1000);
        assert_eq!(idx.len(), 1000);
        // Indices never decrease.
        assert!(sample_indices(260_500).windows(2).all(|p| p[0] <= p[1]));
    }

    #[test]
    fn quantile_type7() {
        let mut x = vec![3.0, 1.0, 2.0, 4.0];
        // index = 1 + 3 * 0.5 = 2.5: halfway between 2 and 3.
        assert_eq!(quantile7(&mut x, 0.5), 2.5);
        assert_eq!(quantile7(&mut x, 1.0), 4.0);
        assert_eq!(quantile7(&mut x, 0.0), 1.0);
    }

    #[test]
    fn a_star_is_masked() {
        // A Gaussian star on a slightly noisy background.
        let (w, h) = (80, 70);
        let mut data = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let (dr, dc) = (r as f64 - 40.0, c as f64 - 50.0);
                let star = (-(dr * dr + dc * dc) / 8.0).exp();
                data.push(0.1 + 0.8 * star + 0.001 * ((r * 7 + c * 13) % 5) as f64);
            }
        }
        let t = Plane::new(w, h, data);
        let p = MaskParams {
            fix_mask: 0.0,
            ..MaskParams::default()
        };
        let m = star_mask(&t, &p, &Progress::new()).unwrap();
        assert!(m.at(40, 50) > 0.99, "{}", m.at(40, 50));
        assert!(m.at(5, 5) < 1e-6, "{}", m.at(5, 5));
        // With a fix mask of 2 the background and the star (the first two regions in
        // scan order) and one more region are left out.
        let p2 = MaskParams::default();
        let m2 = star_mask(&t, &p2, &Progress::new()).unwrap();
        assert!(m2.at(40, 50) < 0.99);
    }
}
