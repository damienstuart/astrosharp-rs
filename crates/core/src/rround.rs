//! R's `round(x, digits)` (R >= 4.0.0, `src/nmath/fround.c`).

/// Rounds like R's `round(x, digits)`: picks the closer of the two neighbouring
/// `digits`-decimal values, breaking exact ties towards an even last digit.
pub fn round_digits(x: f64, digits: i32) -> f64 {
    if x.is_nan() || !x.is_finite() || x == 0.0 {
        return x;
    }
    if digits == 0 {
        return x.round_ties_even();
    }
    let (sgn, x) = if x < 0.0 { (-1.0, -x) } else { (1.0, x) };
    // C's logb(): the binary exponent of x.
    let logb = if x.is_normal() {
        ((x.to_bits() >> 52) & 0x7ff) as f64 - 1023.0
    } else {
        x.log2().floor()
    };
    if digits > 0 && (logb + 0.5) * std::f64::consts::LOG10_2 + digits as f64 > f64::DIGITS as f64 {
        return sgn * x;
    }
    let pow10 = pow_di(10.0, digits);
    let x10 = pow10 * x;
    let i10 = x10.floor();
    let xd = i10 / pow10;
    let xu = x10.ceil() / pow10;
    let du = xu - x;
    let dd = x - xd;
    sgn * if du < dd || (du == dd && i10 % 2.0 == 1.0) {
        xu
    } else {
        xd
    }
}

/// R's `R_pow_di`: integer power by repeated squaring.
fn pow_di(mut x: f64, mut n: i32) -> f64 {
    let mut pow = 1.0;
    if n < 0 {
        n = -n;
        x = 1.0 / x;
    }
    loop {
        if n & 1 != 0 {
            pow *= x;
        }
        n >>= 1;
        if n == 0 {
            break;
        }
        x *= x;
    }
    pow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic() {
        assert_eq!(round_digits(0.123456, 5), 0.12346);
        assert_eq!(round_digits(-0.123456, 5), -0.12346);
        assert_eq!(round_digits(2.5, 0), 2.0);
        assert_eq!(round_digits(3.5, 0), 4.0);
        assert_eq!(pow_di(10.0, 5), 100000.0);
    }
}
