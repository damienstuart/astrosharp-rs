//! sRGB <-> CIE Luv, ported from R's `grDevices::convertColor` so colour images
//! round-trip exactly as in the R app. Note R's D65 white point (x = 0.3137,
//! y = 0.3291) differs slightly from the usual 0.3127/0.3290; the constants
//! below are the values R derives from it.

// The constants are printed with 17 significant digits so they parse to exactly the
// doubles R uses.
#![allow(clippy::excessive_precision)]

use crate::rround::round_digits;

/// `colorspaces$sRGB` matrix `M` (row = R/G/B input, column = X/Y/Z output).
const M: [[f64; 3]; 3] = [
    [
        0.41682134188531689,
        0.21492350440961652,
        0.019538500400874244,
    ],
    [
        0.35657671707797461,
        0.71315343415594923,
        0.11885890569265832,
    ],
    [
        0.17980765358608539,
        0.071923061434434152,
        0.94698697555338307,
    ],
];

/// `solve(M)` (row = X/Y/Z input, column = R/G/B output).
const M_INV: [[f64; 3]; 3] = [
    [
        3.2065205171444644,
        -0.97198254620123203,
        0.055838338593097919,
    ],
    [
        -1.5210417837736561,
        1.8812686516084873,
        -0.20474057484135899,
    ],
    [
        -0.49331084879145598,
        0.041672484599589318,
        1.0609284332688591,
    ],
];

/// `c2to3(white.points[, "D65"])`.
const WHITE: [f64; 3] = [0.95320571254937703, 1.0, 1.0853843816469158];

const EPSILON: f64 = 216.0 / 24389.0;
const KAPPA: f64 = 24389.0 / 27.0;

fn mul(v: [f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    std::array::from_fn(|j| v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j])
}

fn srgb_to_linear(x: f64) -> f64 {
    if x < 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(x: f64) -> f64 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// `convertColor(rgb, from = "sRGB", to = "Luv")` for one pixel.
pub fn srgb_to_luv(rgb: [f64; 3]) -> [f64; 3] {
    let [x, y, z] = mul(rgb.map(srgb_to_linear), &M);
    let yr = y / WHITE[1];
    let denom = x + y * 15.0 + z * 3.0;
    let wdenom = WHITE[0] + WHITE[1] * 15.0 + WHITE[2] * 3.0;
    let (u1, v1) = if denom == 0.0 {
        (1.0, 1.0)
    } else {
        (4.0 * x / denom, 9.0 * y / denom)
    };
    let ur = 4.0 * WHITE[0] / wdenom;
    let vr = 9.0 * WHITE[1] / wdenom;
    let l = if yr <= EPSILON {
        KAPPA * yr
    } else {
        116.0 * yr.powf(1.0 / 3.0) - 16.0
    };
    [l, 13.0 * l * (u1 - ur), 13.0 * l * (v1 - vr)]
}

/// `convertColor(luv, from = "Luv", to = "sRGB")` for one pixel, including R's
/// output trimming (round to 5 decimals, clip to [0, 1]).
pub fn luv_to_srgb(luv: [f64; 3]) -> [f64; 3] {
    let [l, u, v] = luv;
    let xyz = if l == 0.0 {
        [0.0; 3]
    } else {
        let wsum = WHITE[0] + 15.0 * WHITE[1] + 3.0 * WHITE[2];
        let u0 = 4.0 * WHITE[0] / wsum;
        let v0 = 9.0 * WHITE[1] / wsum;
        let y = if l <= KAPPA * EPSILON {
            l / KAPPA
        } else {
            let t = (l + 16.0) / 116.0;
            t * t * t
        };
        let a = (52.0 * l / (u + 13.0 * l * u0) - 1.0) / 3.0;
        let b = -5.0 * y;
        let c = -1.0 / 3.0;
        let d = y * (39.0 * l / (v + 13.0 * l * v0) - 5.0);
        let x = (d - b) / (a - c);
        let z = x * a + b;
        [x, y, z]
    };
    mul(xyz, &M_INV).map(|c| {
        let c = round_digits(linear_to_srgb(c), 5);
        if c.is_nan() { 0.0 } else { c.clamp(0.0, 1.0) }
    })
}
