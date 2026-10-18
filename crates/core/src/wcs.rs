//! Sky coordinates of image pixels from a FITS plate solution (World Coordinate System),
//! as written by Siril, ASTAP or astrometry.net: the gnomonic (TAN) projection, optionally
//! with SIP distortion polynomials.

use crate::FitsHeader;

/// A plate solution mapping image pixels to equatorial coordinates (RA/Dec).
#[derive(Clone, Debug, PartialEq)]
pub struct Wcs {
    /// Reference pixel, in FITS pixel coordinates (1-based, at pixel centres).
    crpix: [f64; 2],
    /// RA and Dec of the reference pixel (degrees).
    crval: [f64; 2],
    /// Linear transformation from pixel offsets to projection-plane coordinates (degrees).
    cd: [[f64; 2]; 2],
    /// Native longitude of the celestial pole (degrees).
    lonpole: f64,
    /// SIP distortion added to the pixel offsets `(u, v)`: `(p, q, coefficient)` terms of
    /// `coefficient * u^p * v^q`, for the A (u) and B (v) polynomials.
    sip: [Vec<(i32, i32, f64)>; 2],
    /// Image height, to turn rows counted from the top into FITS rows.
    height: usize,
    /// Rows are stored top to bottom, so FITS rows count from the top.
    top_down: bool,
}

impl Wcs {
    /// The plate solution in `header` for an image `height` pixels tall, if it has a
    /// usable one: RA/Dec axes with the TAN or TAN-SIP projection.
    pub fn from_header(header: &FitsHeader, height: usize) -> Option<Wcs> {
        Wcs::parse(|key| header.value(key), height, header.top_down())
    }

    fn parse(value: impl Fn(&str) -> Option<String>, height: usize, top_down: bool) -> Option<Wcs> {
        let num = |key: &str| {
            value(key).and_then(|v| v.trim().replace(['D', 'd'], "E").parse::<f64>().ok())
        };
        let has = |keys: &[&str]| keys.iter().any(|k| value(k).is_some());
        let ctype = |key| value(key).map(|v| v.trim().to_string());
        let sip = match (ctype("CTYPE1")?.as_str(), ctype("CTYPE2")?.as_str()) {
            ("RA---TAN", "DEC--TAN") => false,
            ("RA---TAN-SIP", "DEC--TAN-SIP") => true,
            _ => return None,
        };
        if [value("CUNIT1"), value("CUNIT2")]
            .iter()
            .flatten()
            .any(|u| u.trim() != "deg")
        {
            return None;
        }
        let crpix = [num("CRPIX1")?, num("CRPIX2")?];
        let crval = [num("CRVAL1")?, num("CRVAL2")?];
        // CDi_j takes precedence over PCi_j with CDELTi, which takes precedence over the
        // older CDELTi with CROTA2. Missing matrix elements take their default values.
        let cd = if has(&["CD1_1", "CD1_2", "CD2_1", "CD2_2"]) {
            let c = |key| num(key).unwrap_or(0.0);
            [[c("CD1_1"), c("CD1_2")], [c("CD2_1"), c("CD2_2")]]
        } else {
            let cdelt = [num("CDELT1")?, num("CDELT2")?];
            if has(&["PC1_1", "PC1_2", "PC2_1", "PC2_2"]) {
                let pc = |key, default| num(key).unwrap_or(default);
                [
                    [cdelt[0] * pc("PC1_1", 1.0), cdelt[0] * pc("PC1_2", 0.0)],
                    [cdelt[1] * pc("PC2_1", 0.0), cdelt[1] * pc("PC2_2", 1.0)],
                ]
            } else {
                let (sin, cos) = num("CROTA2").unwrap_or(0.0).to_radians().sin_cos();
                [
                    [cdelt[0] * cos, -cdelt[1] * sin],
                    [cdelt[0] * sin, cdelt[1] * cos],
                ]
            }
        };
        let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
        if !det.is_finite() || det == 0.0 {
            return None;
        }
        let poly = |name: &str| -> Vec<(i32, i32, f64)> {
            let Some(order) = num(&format!("{name}_ORDER")).filter(|_| sip) else {
                return Vec::new();
            };
            let order = order as i32;
            (0..=order)
                .flat_map(|p| (0..=order - p).map(move |q| (p, q)))
                .filter_map(|(p, q)| num(&format!("{name}_{p}_{q}")).map(|c| (p, q, c)))
                .filter(|&(_, _, c)| c != 0.0)
                .collect()
        };
        Some(Wcs {
            crpix,
            crval,
            cd,
            // The default for zenithal projections when the reference point isn't the pole.
            lonpole: num("LONPOLE").unwrap_or(180.0),
            sip: [poly("A"), poly("B")],
            height,
            top_down,
        })
    }

    /// RA and Dec (degrees) of the point `(x, y)` in image pixels, counted from the
    /// top-left corner of the image as displayed: pixel `(c, r)` covers `c..c + 1`
    /// horizontally and `r..r + 1` vertically.
    pub fn sky_at(&self, x: f64, y: f64) -> (f64, f64) {
        // FITS pixel coordinates are 1-based at pixel centres, and count rows from the
        // first one stored: the bottom row unless ROWORDER is TOP-DOWN.
        let p1 = x + 0.5;
        let p2 = if self.top_down {
            y + 0.5
        } else {
            self.height as f64 - y + 0.5
        };
        let (u, v) = (p1 - self.crpix[0], p2 - self.crpix[1]);
        let poly = |terms: &[(i32, i32, f64)]| -> f64 {
            terms
                .iter()
                .map(|&(p, q, c)| c * u.powi(p) * v.powi(q))
                .sum()
        };
        let (u, v) = (u + poly(&self.sip[0]), v + poly(&self.sip[1]));
        let px = self.cd[0][0] * u + self.cd[0][1] * v;
        let py = self.cd[1][0] * u + self.cd[1][1] * v;

        // Gnomonic projection, inverted to native spherical coordinates (phi, theta).
        let phi = px.atan2(-py);
        let theta = (180.0 / std::f64::consts::PI).atan2(px.hypot(py));
        // Native to celestial coordinates (Calabretta & Greisen 2002, eq. 2).
        let (sin_t, cos_t) = theta.sin_cos();
        let (sin_d0, cos_d0) = self.crval[1].to_radians().sin_cos();
        let (sin_dp, cos_dp) = (phi - self.lonpole.to_radians()).sin_cos();
        let ra = self.crval[0].to_radians()
            + (-cos_t * sin_dp).atan2(sin_t * cos_d0 - cos_t * sin_d0 * cos_dp);
        let dec = (sin_t * sin_d0 + cos_t * cos_d0 * cos_dp)
            .clamp(-1.0, 1.0)
            .asin();
        (ra.to_degrees().rem_euclid(360.0), dec.to_degrees())
    }
}

/// RA (degrees) as hours, minutes and seconds, rounded to the second: `23h59m47s`.
pub fn format_ra(ra: f64) -> String {
    let s = (ra.rem_euclid(360.0) / 15.0 * 3600.0).round() as i64 % (24 * 3600);
    format!("{:02}h{:02}m{:02}s", s / 3600, s % 3600 / 60, s % 60)
}

/// Dec (degrees) as signed degrees, arcminutes and arcseconds, rounded to the arcsecond:
/// `+66°42'41"`.
pub fn format_dec(dec: f64) -> String {
    let s = (dec.abs() * 3600.0).round() as i64;
    let sign = if dec < 0.0 && s > 0 { '-' } else { '+' };
    format!("{sign}{:02}°{:02}'{:02}\"", s / 3600, s % 3600 / 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plate solution Siril wrote for a 4176x5037 image of NGC 7822 (TAN-SIP, order 3).
    const NGC7822: &str = "CTYPE1='RA---TAN-SIP' CTYPE2='DEC--TAN-SIP' CRPIX1=2090.5 \
        CRPIX2=2505.0 CRVAL1=1.00495931431877 CRVAL2=67.14119479921 LONPOLE=180.0 \
        CDELT1=-0.000311949244482441 CDELT2=0.000311850631330842 PC1_1=0.999999171601768 \
        PC1_2=-0.00128716579246289 PC2_1=0.00135072219572004 PC2_2=0.999999087774359 \
        A_ORDER=3 B_ORDER=3 A_0_0=0.0 A_0_1=0.0 A_0_2=2.47761993453757e-09 \
        A_0_3=-6.56678156446406e-12 A_1_0=0.0 A_1_1=1.59185900629809e-08 \
        A_1_2=-1.6747564521628e-12 A_2_0=-3.09915440250166e-08 A_2_1=-1.82132693759943e-11 \
        A_3_0=-2.95727516795085e-11 B_0_0=0.0 B_0_1=0.0 B_0_2=5.8744642268768e-09 \
        B_0_3=1.19627090605762e-12 B_1_0=0.0 B_1_1=1.78696587326008e-08 \
        B_1_2=-1.08174972500236e-11 B_2_0=2.07153255442646e-08 B_2_1=2.70590365368208e-12 \
        B_3_0=4.18091340293534e-12";
    const HEIGHT: usize = 5037;

    fn parse(cards: &str, height: usize, top_down: bool) -> Option<Wcs> {
        let value = |key: &str| {
            cards.split_whitespace().find_map(|card| {
                let (k, v) = card.split_once('=')?;
                (k == key).then(|| v.trim_matches('\'').to_string())
            })
        };
        Wcs::parse(value, height, top_down)
    }

    /// Angular separation in arcseconds (haversine, accurate for tiny angles).
    fn separation(a: (f64, f64), b: (f64, f64)) -> f64 {
        let (ra1, d1, ra2, d2) = (
            a.0.to_radians(),
            a.1.to_radians(),
            b.0.to_radians(),
            b.1.to_radians(),
        );
        let h = ((d1 - d2) / 2.0).sin().powi(2)
            + d1.cos() * d2.cos() * ((ra1 - ra2) / 2.0).sin().powi(2);
        2.0 * h.sqrt().min(1.0).asin().to_degrees() * 3600.0
    }

    #[test]
    fn matches_astropy() {
        // astropy.wcs all_pix2world at FITS pixel (x + 0.5, HEIGHT - y + 0.5): image corners,
        // centre, Siril's screenshot position and two arbitrary points.
        let reference = [
            (0.5, 0.5, 2.7419059226275597, 67.92075837342526),
            (4175.5, 0.5, 359.2773140494266, 67.922607223882),
            (0.5, 5036.5, 2.6269324418159754, 66.35114040853666),
            (4175.5, 5036.5, 359.3806938758409, 66.35275205897187),
            (2088.0, 2518.5, 1.0065801523281759, 67.1455598536364),
            (3432.5, 3900.5, 359.9443746685778, 66.7115886242145),
            (100.25, 4000.75, 2.5709133690358916, 66.67475559310314),
            (4000.0, 250.0, 359.4277589374468, 67.84622361179693),
        ];
        let wcs = parse(NGC7822, HEIGHT, false).expect("valid plate solution");
        for (x, y, ra, dec) in reference {
            let got = wcs.sky_at(x, y);
            let error = separation(got, (ra, dec));
            assert!(
                error < 1e-3,
                "({x}, {y}): {got:?} is {error}\" from astropy"
            );
        }
    }

    #[test]
    fn reference_pixel_is_crval() {
        let wcs = parse(NGC7822, HEIGHT, false).unwrap();
        // FITS pixel (2090.5, 2505) is x 2090, y HEIGHT - 2505 + 0.5 in display coordinates.
        let got = wcs.sky_at(2090.0, HEIGHT as f64 - 2505.0 + 0.5);
        assert!(separation(got, (1.00495931431877, 67.14119479921)) < 1e-6);
    }

    #[test]
    fn matrix_conventions_agree() {
        // The same linear transformation as a CD matrix, as PC + CDELT and as CROTA2 + CDELT.
        let (cdelt1, cdelt2, rot) = (-0.0003, 0.0003, 30.0_f64);
        let (s, c) = rot.to_radians().sin_cos();
        let base = "CTYPE1=RA---TAN CTYPE2=DEC--TAN CRPIX1=500 CRPIX2=400 CRVAL1=150 CRVAL2=-30";
        let cd = format!(
            "{base} CD1_1={} CD1_2={} CD2_1={} CD2_2={}",
            cdelt1 * c,
            -cdelt2 * s,
            cdelt1 * s,
            cdelt2 * c
        );
        let pc = format!(
            "{base} CDELT1={cdelt1} CDELT2={cdelt2} PC1_1={c} PC1_2={} PC2_1={} PC2_2={c}",
            -s * cdelt2 / cdelt1,
            s * cdelt1 / cdelt2
        );
        let crota = format!("{base} CDELT1={cdelt1} CDELT2={cdelt2} CROTA2={rot}");
        let [cd, pc, crota] = [cd, pc, crota].map(|cards| parse(&cards, 800, false).unwrap());
        for (x, y) in [(0.0, 0.0), (999.0, 799.0), (123.4, 567.8)] {
            let want = cd.sky_at(x, y);
            assert!(separation(pc.sky_at(x, y), want) < 1e-6);
            assert!(separation(crota.sky_at(x, y), want) < 1e-6);
        }
    }

    #[test]
    fn top_down_rows() {
        let bottom_up = parse(NGC7822, HEIGHT, false).unwrap();
        let top_down = parse(NGC7822, HEIGHT, true).unwrap();
        // Display row y of a top-down file is FITS row y + 0.5, which a bottom-up file
        // displays at HEIGHT - y.
        let (x, y) = (1234.5, 678.25);
        let got = top_down.sky_at(x, y);
        let want = bottom_up.sky_at(x, HEIGHT as f64 - y);
        assert!(separation(got, want) < 1e-6);
    }

    #[test]
    fn rejects_unusable_solutions() {
        // Pointing coordinates only, as in an unsolved Siril stack.
        assert!(parse("RA=333.76 DEC=73.55", 100, false).is_none());
        // Projections other than TAN.
        let sin = "CTYPE1=RA---SIN CTYPE2=DEC--SIN CRPIX1=1 CRPIX2=1 CRVAL1=0 CRVAL2=0 \
                   CDELT1=-1e-4 CDELT2=1e-4";
        assert!(parse(sin, 100, false).is_none());
        // A singular matrix.
        let singular = "CTYPE1=RA---TAN CTYPE2=DEC--TAN CRPIX1=1 CRPIX2=1 CRVAL1=0 CRVAL2=0 \
                        CD1_1=1e-4 CD1_2=1e-4 CD2_1=1e-4 CD2_2=1e-4";
        assert!(parse(singular, 100, false).is_none());
    }

    #[test]
    fn formatting() {
        // Siril's readout for the screenshot position: 23h59m47s, +66°42'41".
        assert_eq!(format_ra(359.9443746685778), "23h59m47s");
        assert_eq!(format_dec(66.7115), "+66°42'41\"");
        assert_eq!(format_ra(359.99999), "00h00m00s");
        assert_eq!(format_ra(15.0 + 59.6 / 3600.0 * 15.0), "01h01m00s");
        assert_eq!(format_dec(-5.25), "-05°15'00\"");
        assert_eq!(format_dec(-0.0001), "+00°00'00\"");
    }
}
