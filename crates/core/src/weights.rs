//! Embedded model weights, exported from the AstroSharp `.RDS` files by
//! `tools/export_weights.R`. Models are parsed lazily on first use.

use std::sync::OnceLock;

use crate::Error;
use crate::model::Mlp;

/// Smallest PSF FWHM with a trained model; models exist every [`PSF_STEP`] up to [`PSF_MAX`].
pub const PSF_MIN: f64 = 1.0;
pub const PSF_MAX: f64 = 8.0;
pub const PSF_STEP: f64 = 0.25;
const PSF_COUNT: usize = 29;

macro_rules! psf_blobs {
    ($($name:literal),* $(,)?) => {
        [$(include_bytes!(concat!("../weights/psf_", $name, ".bin")) as &[u8]),*]
    };
}

// File names use R's shortest number formatting, as in the app's paste0("81_1_FWHM_", v, ".RDS").
static PSF_BLOBS: [&[u8]; PSF_COUNT] = psf_blobs!(
    "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5", "2.75", "3", "3.25", "3.5", "3.75", "4",
    "4.25", "4.5", "4.75", "5", "5.25", "5.5", "5.75", "6", "6.25", "6.5", "6.75", "7", "7.25",
    "7.5", "7.75", "8",
);
static ASTROCLEAN_BLOB: &[u8] = include_bytes!("../weights/astroclean.bin");

static PSF_MODELS: [OnceLock<Mlp>; PSF_COUNT] = [const { OnceLock::new() }; PSF_COUNT];
static ASTROCLEAN_MODEL: OnceLock<Mlp> = OnceLock::new();

fn parse(blob: &[u8]) -> Mlp {
    Mlp::from_bytes(blob).expect("embedded weights are valid")
}

/// Index of the PSF model for `fwhm`, if it lies on the 0.25 grid between 1 and 8.
pub fn psf_index(fwhm: f64) -> Option<usize> {
    let k = (fwhm - PSF_MIN) / PSF_STEP;
    let i = k.round();
    ((k - i).abs() < 1e-6 && (0.0..PSF_COUNT as f64).contains(&i)).then_some(i as usize)
}

/// The PSF sharpening model trained for the given FWHM (1–8 in steps of 0.25).
pub fn psf(fwhm: f64) -> Result<&'static Mlp, Error> {
    let i = psf_index(fwhm).ok_or_else(|| {
        Error::InvalidParams(format!(
            "PSF FWHM must be between {PSF_MIN} and {PSF_MAX} in steps of {PSF_STEP}, got {fwhm}"
        ))
    })?;
    Ok(PSF_MODELS[i].get_or_init(|| parse(PSF_BLOBS[i])))
}

/// The AstroClean denoising model (`AstroCleanv2NoBeta.RDS`).
pub fn astroclean() -> &'static Mlp {
    ASTROCLEAN_MODEL.get_or_init(|| parse(ASTROCLEAN_BLOB))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_models_parse_with_expected_shape() {
        for i in 0..PSF_COUNT {
            let m = psf(PSF_MIN + i as f64 * PSF_STEP).unwrap();
            let shape: Vec<_> = m.layers.iter().map(|l| (l.inputs, l.outputs)).collect();
            assert_eq!(shape, [(81, 82), (82, 49), (49, 25), (25, 9), (9, 1)]);
        }
        assert_eq!(astroclean().inputs(), 81);
    }

    #[test]
    fn psf_grid() {
        assert_eq!(psf_index(1.0), Some(0));
        assert_eq!(psf_index(3.25), Some(9));
        assert_eq!(psf_index(8.0), Some(28));
        assert_eq!(psf_index(3.1), None);
        assert_eq!(psf_index(8.25), None);
        assert_eq!(psf_index(0.75), None);
    }
}
