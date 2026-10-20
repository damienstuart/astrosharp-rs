//! The regularized incomplete beta function, R's `pbeta(x, a, b)`, which the R app uses to
//! shape the star mask.
//!
//! Ported from the Cephes Math Library (`incbet.c`, Copyright 1984, 1995, 2000 by
//! Stephen L. Moshier, BSD licensed as distributed with SciPy). Accurate to about 1e-13;
//! R's TOMS 708 implementation is not reused because it is GPL.

/// 2^-53.
const MACHEP: f64 = 1.110_223_024_625_156_5e-16;
const MAXLOG: f64 = 7.097_827_128_933_84e2;
const MINLOG: f64 = -7.083_964_185_322_641e2;
const MAXGAM: f64 = 171.624_376_956_302_7;
const BIG: f64 = 4.503_599_627_370_496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

/// `pbeta(x, a, b)` for fixed shape parameters, with the beta function precomputed.
#[derive(Clone, Copy, Debug)]
pub struct PBeta {
    a: f64,
    b: f64,
    /// `1 / B(a, b)` when `a + b` is small enough for the direct formula.
    inv_beta: f64,
    /// `ln B(a, b)`.
    lbeta: f64,
}

impl PBeta {
    /// Shape parameters must be positive and finite.
    pub fn new(a: f64, b: f64) -> PBeta {
        assert!(
            a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite(),
            "beta shape parameters must be positive, got {a} and {b}"
        );
        let inv_beta = if a + b < MAXGAM {
            1.0 / (libm::tgamma(a) * libm::tgamma(b) / libm::tgamma(a + b))
        } else {
            0.0
        };
        PBeta {
            a,
            b,
            inv_beta,
            lbeta: libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b),
        }
    }

    /// The probability `P(X <= x)` for `X ~ Beta(a, b)`: 0 for `x <= 0`, 1 for `x >= 1`.
    pub fn at(&self, x: f64) -> f64 {
        if x.is_nan() {
            return x;
        }
        if x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let (aa, bb) = (self.a, self.b);
        if bb * x <= 1.0 && x <= 0.95 {
            return self.pseries(aa, bb, x);
        }
        let w = 1.0 - x;
        // Swap a and b when x is above the mean.
        let flip = x > aa / (aa + bb);
        let (a, b, xc, x) = if flip { (bb, aa, x, w) } else { (aa, bb, w, x) };
        let t = if flip && b * x <= 1.0 && x <= 0.95 {
            self.pseries(a, b, x)
        } else {
            // Choose the expansion that converges faster.
            let y = x * (a + b - 2.0) - (a - 1.0);
            let w = if y < 0.0 {
                incbcf(a, b, x)
            } else {
                incbd(a, b, x) / xc
            };
            // Multiply by x^a (1-x)^b Gamma(a+b) / (a Gamma(a) Gamma(b)).
            let y = a * x.ln();
            let t = b * xc.ln();
            if a + b < MAXGAM && y.abs() < MAXLOG && t.abs() < MAXLOG {
                let mut t = xc.powf(b);
                t *= x.powf(a);
                t /= a;
                t *= w;
                t * self.inv_beta
            } else {
                let mut y = y;
                y += t - self.lbeta;
                y += (w / a).ln();
                if y < MINLOG { 0.0 } else { y.exp() }
            }
        };
        if flip {
            if t <= MACHEP { 1.0 - MACHEP } else { 1.0 - t }
        } else {
            t
        }
    }

    /// Power series, for small `b * x` with `x` not too close to 1.
    fn pseries(&self, a: f64, b: f64, x: f64) -> f64 {
        let ai = 1.0 / a;
        let mut u = (1.0 - b) * x;
        let mut v = u / (a + 1.0);
        let t1 = v;
        let mut t = u;
        let mut n = 2.0;
        let mut s = 0.0;
        let z = MACHEP * ai;
        while v.abs() > z {
            u = (n - b) * x / n;
            t *= u;
            v = t / (a + n);
            s += v;
            n += 1.0;
        }
        s += t1;
        s += ai;
        let u = a * x.ln();
        if a + b < MAXGAM && u.abs() < MAXLOG {
            s * self.inv_beta * x.powf(a)
        } else {
            let t = -self.lbeta + u + s.ln();
            if t < MINLOG { 0.0 } else { t.exp() }
        }
    }
}

/// R's `pbeta(x, a, b)` (lower tail, not logged).
pub fn pbeta(x: f64, a: f64, b: f64) -> f64 {
    PBeta::new(a, b).at(x)
}

/// Continued fraction expansion #1.
fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let (mut k1, mut k2, mut k3, mut k4) = (a, a + b, a, a + 1.0);
    let (mut k5, mut k6, mut k7, mut k8) = (1.0, b - 1.0, a + 1.0, a + 2.0);
    continued_fraction(x, |step| {
        if step {
            k1 += 1.0;
            k2 += 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 -= 1.0;
            k7 += 2.0;
            k8 += 2.0;
        }
        (k1, k2, k3, k4, k5, k6, k7, k8)
    })
}

/// Continued fraction expansion #2.
fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let z = x / (1.0 - x);
    let (mut k1, mut k2, mut k3, mut k4) = (a, b - 1.0, a, a + 1.0);
    let (mut k5, mut k6, mut k7, mut k8) = (1.0, a + b, a + 1.0, a + 2.0);
    continued_fraction(z, |step| {
        if step {
            k1 += 1.0;
            k2 -= 1.0;
            k3 += 2.0;
            k4 += 2.0;
            k5 += 1.0;
            k6 += 1.0;
            k7 += 2.0;
            k8 += 2.0;
        }
        (k1, k2, k3, k4, k5, k6, k7, k8)
    })
}

/// The evaluation loop shared by [`incbcf`] and [`incbd`]; `ks(step)` advances the
/// coefficients (when `step`) and returns them.
fn continued_fraction(
    x: f64,
    mut ks: impl FnMut(bool) -> (f64, f64, f64, f64, f64, f64, f64, f64),
) -> f64 {
    let (mut pkm2, mut qkm2, mut pkm1, mut qkm1) = (0.0, 1.0, 1.0, 1.0);
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;
    for n in 0..300 {
        let (k1, k2, k3, k4, k5, k6, k7, k8) = ks(n > 0);
        let xk = -(x * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        (pkm2, pkm1, qkm2, qkm1) = (pkm1, pk, qkm1, qk);

        let xk = (x * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        (pkm2, pkm1, qkm2, qkm1) = (pkm1, pk, qkm1, qk);

        if qk != 0.0 {
            r = pk / qk;
        }
        let t = if r != 0.0 {
            let t = ((ans - r) / r).abs();
            ans = r;
            t
        } else {
            1.0
        };
        if t < thresh {
            break;
        }
        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges() {
        assert_eq!(pbeta(0.0, 9.0, 100.0), 0.0);
        assert_eq!(pbeta(-1.0, 9.0, 100.0), 0.0);
        assert_eq!(pbeta(1.0, 9.0, 100.0), 1.0);
        assert_eq!(pbeta(2.0, 9.0, 100.0), 1.0);
        assert!(pbeta(f64::NAN, 9.0, 100.0).is_nan());
    }

    #[test]
    fn uniform_is_the_identity() {
        for x in [1e-9, 0.1, 0.25, 0.5, 0.9, 0.999] {
            assert!((pbeta(x, 1.0, 1.0) - x).abs() < 1e-15, "{x}");
        }
    }

    #[test]
    fn symmetry() {
        for (a, b) in [
            (9.0, 100.0),
            (0.5, 0.5),
            (2.0, 3.0),
            (1e-4, 1e-4),
            (50.0, 500.0),
        ] {
            for x in [0.01, 0.08, 0.3, 0.5, 0.77] {
                let s = pbeta(x, a, b) + pbeta(1.0 - x, b, a);
                assert!((s - 1.0).abs() < 1e-12, "{a} {b} {x}: {s}");
            }
        }
    }

    #[test]
    fn known_values() {
        // Closed form for a = 2, b = 1: x^2.
        assert!((pbeta(0.3, 2.0, 1.0) - 0.09).abs() < 1e-15);
        // a = 1, b = 2: 1 - (1 - x)^2.
        assert!((pbeta(0.3, 1.0, 2.0) - 0.51).abs() < 1e-15);
        // a = b = 0.5: (2 / pi) asin(sqrt(x)).
        let x: f64 = 0.2;
        let exact = 2.0 / std::f64::consts::PI * x.sqrt().asin();
        assert!((pbeta(x, 0.5, 0.5) - exact).abs() < 1e-14);
    }
}
