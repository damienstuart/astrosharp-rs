//! Ports of the CImg filters (CImg 3.3.4, as shipped with R's imager 1.0.8) that the
//! R app's star mask uses: `isoblur` (Van Vliet / Deriche recursive Gaussian with Neumann
//! boundaries) and `imhessian`.
//!
//! imager maps an R matrix's rows to CImg's x axis, so CImg's x runs down a [`Plane`]'s
//! columns (vertically) and y along its rows. Arithmetic follows CImg's source order with
//! plain IEEE operations (no fused multiply-add), so results match a non-FMA build of
//! imager bit for bit.

use rayon::prelude::*;

use crate::Plane;

/// Columns per task in the vertical pass, gathered into a contiguous buffer.
const BAND: usize = 32;

/// A recursive Gaussian along one axis.
#[derive(Clone, Copy, Debug)]
enum Kernel {
    VanVliet(VanVliet),
    Deriche(Deriche),
}

impl Kernel {
    /// The filter `vanvliet()` applies for `sigma` (single precision, as CImg takes it), or
    /// `None` if it leaves the data unchanged.
    fn new(sigma: f32) -> Option<Kernel> {
        let s = sigma as f64;
        if s < 0.1f32 as f64 {
            None
        } else if s < 0.5f32 as f64 {
            Some(Kernel::Deriche(Deriche::new(s)))
        } else {
            Some(Kernel::VanVliet(VanVliet::new(s)))
        }
    }

    /// Filters one contiguous line in place; `scratch` is a reusable buffer.
    fn apply(&self, x: &mut [f64], scratch: &mut Vec<f64>) {
        match self {
            Kernel::VanVliet(k) => k.apply(x),
            Kernel::Deriche(k) => k.apply(x, scratch),
        }
    }
}

/// Coefficients of CImg's `vanvliet()` (order 0) with Triggs' boundary matrix
/// (`_cimg_recursive_apply`).
#[derive(Clone, Copy, Debug)]
struct VanVliet {
    /// `[B, -b1, -b2, -b3]`.
    filter: [f64; 4],
    m: [f64; 9],
}

impl VanVliet {
    fn new(nnsigma: f64) -> VanVliet {
        let (m0, m1, m2) = (1.16680, 1.10783, 1.40586);
        let (m1sq, m2sq) = (m1 * m1, m2 * m2);
        let q = if nnsigma < 3.556 {
            -0.2568 + 0.5784 * nnsigma + 0.0561 * nnsigma * nnsigma
        } else {
            2.5091 + 0.9804 * (nnsigma - 3.556)
        };
        let qsq = q * q;
        let scale = (m0 + q) * (m1sq + m2sq + 2.0 * m1 * q + qsq);
        let b1 = -q * (2.0 * m0 * m1 + m1sq + m2sq + (2.0 * m0 + 4.0 * m1) * q + 3.0 * qsq) / scale;
        let b2 = qsq * (m0 + 2.0 * m1 + 3.0 * q) / scale;
        let b3 = -qsq * q / scale;
        let b = (m0 * (m1sq + m2sq)) / scale;
        let filter = [b, -b1, -b2, -b3];

        let [_, a1, a2, a3] = filter;
        let scale_m = 1. / ((1. + a1 - a2 + a3) * (1. - a1 - a2 - a3) * (1. + a2 + (a1 - a3) * a3));
        let m = [
            scale_m * (-a3 * a1 + 1. - a3 * a3 - a2),
            scale_m * (a3 + a1) * (a2 + a3 * a1),
            scale_m * a3 * (a1 + a3 * a2),
            scale_m * (a1 + a3 * a2),
            -scale_m * (a2 - 1.) * (a2 + a3 * a1),
            -scale_m * a3 * (a3 * a1 + a3 * a3 + a2 - 1.),
            scale_m * (a3 * a1 + a2 + a1 * a1 - a2 * a2),
            scale_m * (a1 * a2 + a3 * a2 * a2 - a1 * a3 * a3 - a3 * a3 * a3 - a3 * a2 + a3),
            scale_m * a3 * (a1 + a3 * a2),
        ];
        VanVliet { filter, m }
    }

    /// `_cimg_recursive_apply` for order 0 with Neumann boundaries: a causal pass, Triggs'
    /// end condition, then an anti-causal pass.
    fn apply(&self, x: &mut [f64]) {
        let n = x.len();
        let [sumsq, a1, a2, a3] = self.filter;
        let sum = sumsq * sumsq;
        let m = &self.m;
        let iplus = x[n - 1];
        let (mut v1, mut v2, mut v3) = (x[0] / sumsq, x[0] / sumsq, x[0] / sumsq);
        for xi in x.iter_mut() {
            let mut v = *xi;
            v += v1 * a1;
            v += v2 * a2;
            v += v3 * a3;
            *xi = v;
            (v3, v2, v1) = (v2, v1, v);
        }
        let uplus = iplus / (1. - a1 - a2 - a3);
        let vplus = uplus / (1. - a1 - a2 - a3);
        let (unp, unp1, unp2) = (v1 - uplus, v2 - uplus, v3 - uplus);
        let w0 = (m[0] * unp + m[1] * unp1 + m[2] * unp2 + vplus) * sum;
        let w1 = (m[3] * unp + m[4] * unp1 + m[5] * unp2 + vplus) * sum;
        let w2 = (m[6] * unp + m[7] * unp1 + m[8] * unp2 + vplus) * sum;
        x[n - 1] = w0;
        (v1, v2, v3) = (w0, w1, w2);
        for xi in x[..n - 1].iter_mut().rev() {
            let mut v = *xi * sum;
            v += v1 * a1;
            v += v2 * a2;
            v += v3 * a3;
            *xi = v;
            (v3, v2, v1) = (v2, v1, v);
        }
    }
}

/// Coefficients of CImg's `deriche()` (order 0), which `vanvliet()` falls back to for
/// sigma below 0.5.
#[derive(Clone, Copy, Debug)]
struct Deriche {
    a0: f64,
    a1: f64,
    a2: f64,
    a3: f64,
    b1: f64,
    b2: f64,
    coefp: f64,
    coefn: f64,
}

impl Deriche {
    fn new(nnsigma: f64) -> Deriche {
        let alpha = 1.695f32 as f64 / nnsigma;
        let ema = (-alpha).exp();
        let ema2 = (-2.0 * alpha).exp();
        let b1 = -2.0 * ema;
        let b2 = ema2;
        let k = (1.0 - ema) * (1.0 - ema) / (1.0 + 2.0 * alpha * ema - ema2);
        let (a0, a1, a2, a3) = (
            k,
            k * (alpha - 1.0) * ema,
            k * (alpha + 1.0) * ema,
            -k * ema2,
        );
        Deriche {
            a0,
            a1,
            a2,
            a3,
            b1,
            b2,
            coefp: (a0 + a1) / (1.0 + b1 + b2),
            coefn: (a2 + a3) / (1.0 + b1 + b2),
        }
    }

    /// `_cimg_deriche_apply` with Neumann boundaries.
    fn apply(&self, x: &mut [f64], y: &mut Vec<f64>) {
        let Deriche {
            a0,
            a1,
            a2,
            a3,
            b1,
            b2,
            coefp,
            coefn,
        } = *self;
        let n = x.len();
        y.clear();
        y.reserve(n);
        let mut xp = x[0];
        let (mut yb, mut yp) = (coefp * xp, coefp * xp);
        for &xc in x.iter() {
            let yc = a0 * xc + a1 * xp - b1 * yp - b2 * yb;
            y.push(yc);
            (xp, yb, yp) = (xc, yp, yc);
        }
        let (mut xn, mut xa) = (x[n - 1], x[n - 1]);
        let (mut yn, mut ya) = (coefn * xn, coefn * xn);
        for (xi, &yi) in x.iter_mut().zip(y.iter()).rev() {
            let xc = *xi;
            let yc = a2 * xn + a3 * xa - b1 * yn - b2 * ya;
            (xa, xn, ya, yn) = (xn, xc, yn, yc);
            *xi = yi + yc;
        }
    }
}

/// imager's `isoblur(im, sigma)` (Gaussian, Neumann boundaries): CImg's `blur()`, which
/// filters along x (down the plane's columns), then along y (along its rows). Sigma is
/// rounded to single precision as imager passes it; below 0.1 the plane is unchanged.
pub fn isoblur(p: &mut Plane, sigma: f64) {
    assert!(sigma >= 0.0, "blur sigma must not be negative");
    let Some(kernel) = Kernel::new(sigma as f32) else {
        return;
    };
    let (w, h) = (p.width, p.height);
    if h > 1 {
        blur_columns(p, &kernel);
    }
    if w > 1 {
        p.data
            .par_chunks_mut(w)
            .for_each_init(Vec::new, |scratch, row| kernel.apply(row, scratch));
    }
}

/// Filters every column. Bands of columns are gathered into contiguous buffers so each
/// column is filtered as one line, then written back.
fn blur_columns(p: &mut Plane, kernel: &Kernel) {
    let (w, h) = (p.width, p.height);
    let nbands = w.div_ceil(BAND);
    let mut bands: Vec<Vec<&mut [f64]>> = (0..nbands).map(|_| Vec::with_capacity(h)).collect();
    for row in p.data.chunks_mut(w) {
        for (b, seg) in row.chunks_mut(BAND).enumerate() {
            bands[b].push(seg);
        }
    }
    bands.into_par_iter().for_each(|mut rows| {
        let bw = rows[0].len();
        let mut buf = vec![0.0; bw * h];
        for (r, seg) in rows.iter().enumerate() {
            for (j, &v) in seg.iter().enumerate() {
                buf[j * h + r] = v;
            }
        }
        let mut scratch = Vec::new();
        for col in buf.chunks_mut(h) {
            kernel.apply(col, &mut scratch);
        }
        for (r, seg) in rows.iter_mut().enumerate() {
            for (j, v) in seg.iter_mut().enumerate() {
                *v = buf[j * h + r];
            }
        }
    });
}

/// The Hessian `imhessian(g)` computes, as `(xx, xy, yy)` planes. Exposed for tests; the
/// star mask uses [`fold_hessian_det`].
pub fn hessian(g: &Plane) -> (Plane, Plane, Plane) {
    let (w, h) = (g.width, g.height);
    let mut xx = Plane::zeros(w, h);
    let mut xy = Plane::zeros(w, h);
    let mut yy = Plane::zeros(w, h);
    for r in 0..h {
        for c in 0..w {
            let (dxx, dxy, dyy) = hessian_at(g, r, c);
            xx.set(r, c, dxx);
            xy.set(r, c, dxy);
            yy.set(r, c, dyy);
        }
    }
    (xx, xy, yy)
}

/// CImg's `get_hessian("xxxyyy")` at one pixel. Second differences use one-sided forms at
/// the edges; the mixed derivative uses clamped neighbours.
#[inline]
fn hessian_at(g: &Plane, r: usize, c: usize) -> (f64, f64, f64) {
    let (w, h) = (g.width, g.height);
    let at = |r: usize, c: usize| g.data[r * w + c];
    let x = at(r, c);
    // Along x (rows of the plane); CImg returns 0 when the axis has a single pixel.
    let xx = if h == 1 {
        0.0
    } else if r == 0 {
        at(1, c) - x
    } else if r == h - 1 {
        at(h - 2, c) - x
    } else {
        at(r + 1, c) + at(r - 1, c) - 2.0 * x
    };
    let yy = if w == 1 {
        0.0
    } else if c == 0 {
        at(r, 1) - x
    } else if c == w - 1 {
        at(r, w - 2) - x
    } else {
        at(r, c + 1) + at(r, c - 1) - 2.0 * x
    };
    let xy = if h == 1 || w == 1 {
        0.0
    } else {
        let (rp, rn) = (r.saturating_sub(1), (r + 1).min(h - 1));
        let (cp, cn) = (c.saturating_sub(1), (c + 1).min(w - 1));
        // (Inn + Ipp - Inp - Ipn) / 4, where CImg's first index is x (our row).
        (at(rn, cn) + at(rp, cp) - at(rn, cp) - at(rp, cn)) / 4.0
    };
    (xx, xy, yy)
}

/// Folds the scale-normalised determinant of the Hessian of `g` into `d`:
/// `d = max(d, s2 * (xx * yy - xy^2))`, as the R app's `hessdet` and `parmax` do.
pub fn fold_hessian_det(g: &Plane, s2: f64, d: &mut Plane) {
    assert_eq!((g.width, g.height), (d.width, d.height));
    let w = g.width;
    d.data.par_chunks_mut(w).enumerate().for_each(|(r, drow)| {
        for (c, dv) in drow.iter_mut().enumerate() {
            let (xx, xy, yy) = hessian_at(g, r, c);
            let v = s2 * (xx * yy - xy * xy);
            if v > *dv {
                *dv = v;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_planes_stay_constant() {
        for sigma in [0.3, 0.5, 2.0, 5.0, 20.0] {
            let mut p = Plane::new(37, 23, vec![0.25; 37 * 23]);
            isoblur(&mut p, sigma);
            for &v in &p.data {
                assert!((v - 0.25).abs() < 1e-12, "sigma {sigma}: {v}");
            }
        }
    }

    #[test]
    fn tiny_sigma_is_a_no_op() {
        let data: Vec<f64> = (0..100).map(|i| (i % 7) as f64).collect();
        let mut p = Plane::new(10, 10, data.clone());
        isoblur(&mut p, 0.05);
        assert_eq!(p.data, data);
    }

    #[test]
    fn blur_preserves_mass_roughly_and_spreads_a_point() {
        let mut p = Plane::zeros(41, 41);
        p.set(20, 20, 1.0);
        isoblur(&mut p, 3.0);
        let total: f64 = p.data.iter().sum();
        assert!((total - 1.0).abs() < 1e-3, "{total}");
        // Symmetric about the centre in both directions.
        assert!((p.at(17, 20) - p.at(23, 20)).abs() < 1e-6);
        assert!((p.at(20, 17) - p.at(20, 23)).abs() < 1e-6);
        assert!((p.at(17, 20) - p.at(20, 17)).abs() < 1e-6);
    }

    #[test]
    fn deriche_path_smooths() {
        let mut p = Plane::zeros(21, 21);
        p.set(10, 10, 1.0);
        isoblur(&mut p, 0.3);
        assert!(p.at(10, 10) < 1.0 && p.at(10, 11) > 0.0);
    }

    #[test]
    fn hessian_of_a_quadratic() {
        // g = r^2 + 3 c^2 + 2 r c: xx = 2, yy = 6, xy = 2 in the interior.
        let (w, h) = (6, 5);
        let data = (0..h)
            .flat_map(|r| (0..w).map(move |c| (r * r + 3 * c * c + 2 * r * c) as f64))
            .collect();
        let g = Plane::new(w, h, data);
        let (xx, xy, yy) = hessian(&g);
        assert_eq!((xx.at(2, 2), xy.at(2, 2), yy.at(2, 2)), (2.0, 2.0, 6.0));
        // One-sided at the edges: g[1] - g[0] along x at r = 0, c = 0.
        assert_eq!(xx.at(0, 0), 1.0);
        let mut d = Plane::new(w, h, vec![f64::NEG_INFINITY; w * h]);
        fold_hessian_det(&g, 4.0, &mut d);
        assert_eq!(d.at(2, 2), 4.0 * (2.0 * 6.0 - 4.0));
    }
}
