//! FITS reading and writing for images in the primary HDU, as Siril writes them: grey
//! (2 axes) or RGB (3 planes), integer or floating point samples, uncompressed.
//!
//! The source header is kept so a processed image can be written back with the same
//! metadata (observation details, plate solution, history, ...): structural keywords
//! are regenerated, everything else is copied verbatim, and any HDUs after the primary
//! image are passed through unchanged.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{BitDepth, Error, Image};

const BLOCK: usize = 2880;
const CARD: usize = 80;

/// Metadata of a FITS source, kept to write the processed image back faithfully.
#[derive(Clone, Debug, PartialEq)]
pub struct FitsHeader {
    /// Non-structural cards, verbatim, in their original order.
    cards: Vec<String>,
    /// BITPIX of the source data (0 when there is no FITS source).
    bitpix: i32,
    /// Rows stored top to bottom (`ROWORDER = 'TOP-DOWN'`) instead of FITS's bottom-up.
    top_down: bool,
    /// Physical value that floating point samples were divided by to fit [0, 1].
    float_scale: f64,
    /// HDUs following the primary image, copied through unchanged.
    extensions: Vec<u8>,
}

impl Default for FitsHeader {
    /// Metadata for an image that did not come from a FITS file.
    fn default() -> Self {
        FitsHeader {
            cards: Vec::new(),
            bitpix: 0,
            top_down: false,
            float_scale: 1.0,
            extensions: Vec::new(),
        }
    }
}

impl FitsHeader {
    /// Preserved header cards (80 characters each).
    pub fn cards(&self) -> &[String] {
        &self.cards
    }

    /// The raw value text of the first card with this keyword (strings keep their quotes
    /// stripped).
    pub fn value(&self, key: &str) -> Option<String> {
        let card = self.cards.iter().find(|c| keyword(c) == key)?;
        value_text(card).map(|v| {
            if v.starts_with('\'') {
                unquote(v)
            } else {
                v.to_string()
            }
        })
    }

    /// The bit depth closest to the source data, if there was a FITS source.
    pub fn source_depth(&self) -> Option<BitDepth> {
        match self.bitpix {
            0 => None,
            b if b < 0 => Some(BitDepth::F32),
            _ => Some(BitDepth::U16),
        }
    }

    /// Appends HISTORY cards with `text`, wrapped at 72 characters.
    pub fn add_history(&mut self, text: &str) {
        let chars: Vec<char> = text.chars().filter(char::is_ascii).collect();
        for chunk in chars.chunks(72) {
            self.cards
                .push(pad(format!("HISTORY {}", chunk.iter().collect::<String>())));
        }
    }
}

/// Reads a grey or RGB image from the primary HDU, returning it top row first with
/// samples scaled to [0, 1]: integer data by the range of its type (after BZERO/BSCALE),
/// floating point data as stored unless it is clearly not normalised. NaN becomes 0.
pub fn read_fits(path: impl AsRef<Path>) -> Result<(Image, FitsHeader), Error> {
    let mut r = BufReader::new(File::open(path)?);
    let cards = read_cards(&mut r)?;
    let int = |key: &str| -> Result<i64, Error> {
        find(&cards, key)
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| bad(&format!("missing or invalid {key}")))
    };
    let num = |key: &str, default: f64| {
        find(&cards, key)
            .and_then(|v| v.replace(['D', 'd'], "E").parse::<f64>().ok())
            .unwrap_or(default)
    };
    if find(&cards, "SIMPLE") != Some("T") {
        return Err(bad("not a standard FITS file (SIMPLE must be T)"));
    }
    let bitpix = int("BITPIX")? as i32;
    let naxis = int("NAXIS")?;
    let dims: Vec<usize> = (1..=naxis)
        .map(|i| int(&format!("NAXIS{i}")).map(|n| n as usize))
        .collect::<Result<_, _>>()?;
    let (w, h, planes) = match dims[..] {
        [w, h] => (w, h, 1),
        [w, h, p] if p == 1 || p == 3 => (w, h, p),
        [] => return Err(unsupported("the primary HDU has no image")),
        _ => {
            return Err(unsupported(
                "only grey or 3-channel RGB images are supported",
            ));
        }
    };
    let bytes_per = match bitpix {
        8 => 1,
        16 => 2,
        32 | -32 => 4,
        64 | -64 => 8,
        _ => return Err(bad(&format!("invalid BITPIX {bitpix}"))),
    };
    let (bzero, bscale) = (num("BZERO", 0.0), num("BSCALE", 1.0));
    let top_down = find(&cards, "ROWORDER").is_some_and(|v| unquote(v) == "TOP-DOWN");

    // Physical values, top row first, channels interleaved.
    let mut data = vec![0.0; w * h * planes];
    let mut row = vec![0u8; w * bytes_per];
    for p in 0..planes {
        for file_row in 0..h {
            r.read_exact(&mut row)
                .map_err(|_| bad("image data is truncated"))?;
            let y = if top_down { file_row } else { h - 1 - file_row };
            for (x, s) in row.chunks_exact(bytes_per).enumerate() {
                let raw = match bitpix {
                    8 => s[0] as f64,
                    16 => i16::from_be_bytes([s[0], s[1]]) as f64,
                    32 => i32::from_be_bytes(s.try_into().unwrap()) as f64,
                    64 => i64::from_be_bytes(s.try_into().unwrap()) as f64,
                    -32 => f32::from_be_bytes(s.try_into().unwrap()) as f64,
                    _ => f64::from_be_bytes(s.try_into().unwrap()),
                };
                data[(y * w + x) * planes + p] = bzero + bscale * raw;
            }
        }
    }
    let skip = (BLOCK - (w * h * planes * bytes_per) % BLOCK) % BLOCK;
    std::io::copy(&mut (&mut r).take(skip as u64), &mut std::io::sink())?;
    let mut extensions = Vec::new();
    r.read_to_end(&mut extensions)?;

    // Scale to [0, 1].
    let mut float_scale = 1.0;
    let divisor = match bitpix {
        8 => 255.0,
        16 => 65535.0,
        32 => u32::MAX as f64,
        64 => u64::MAX as f64,
        _ => {
            // Siril stores floats in [0, 1]; some programs store ADU-like ranges.
            let max = data
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0, f64::max);
            if max > 1.5 {
                float_scale = if max <= 65535.0 { 65535.0 } else { max };
            }
            float_scale
        }
    };
    for v in &mut data {
        *v = if v.is_nan() { 0.0 } else { *v / divisor };
    }
    let channels = if planes == 3 { 3 } else { 1 };
    let header = FitsHeader {
        cards: cards
            .into_iter()
            .filter(|c| !is_structural(keyword(c)))
            .collect(),
        bitpix,
        top_down,
        float_scale,
        extensions,
    };
    Ok((Image::new(w, h, channels, data), header))
}

/// Writes `img` as a FITS file: 16-bit unsigned (BZERO 32768) or 32-bit float samples,
/// in the row order of `header`'s source. `header` supplies the preserved cards and any
/// extension HDUs; without one, a minimal bottom-up header is written.
pub fn write_fits(
    path: impl AsRef<Path>,
    img: &Image,
    depth: BitDepth,
    header: Option<&FitsHeader>,
) -> Result<(), Error> {
    let default = FitsHeader::default();
    let src = header.unwrap_or(&default);
    let bitpix = match depth {
        BitDepth::U16 => 16,
        BitDepth::F32 => -32,
        BitDepth::U8 => return Err(unsupported("8-bit FITS output")),
    };
    let (w, h) = (img.width, img.height);
    let mut cards = vec![
        logical_card("SIMPLE", true, "file does conform to FITS standard"),
        int_card("BITPIX", bitpix, "number of bits per data pixel"),
        int_card(
            "NAXIS",
            if img.is_color() { 3 } else { 2 },
            "number of data axes",
        ),
        int_card("NAXIS1", w as i64, "length of data axis 1"),
        int_card("NAXIS2", h as i64, "length of data axis 2"),
    ];
    if img.is_color() {
        cards.push(int_card("NAXIS3", 3, "length of data axis 3"));
    }
    cards.push(logical_card(
        "EXTEND",
        true,
        "FITS dataset may contain extensions",
    ));
    if depth == BitDepth::U16 {
        cards.push(int_card(
            "BZERO",
            32768,
            "offset data range to that of unsigned short",
        ));
        cards.push(int_card("BSCALE", 1, "default scaling factor"));
    }
    let date = string_card("DATE", &utc_now(), "UTC date that FITS file was created");
    let mut dated = false;
    for card in &src.cards {
        if keyword(card) == "DATE" {
            cards.push(date.clone());
            dated = true;
        } else {
            cards.push(card.clone());
        }
    }
    if !dated {
        cards.push(date);
    }
    if src.bitpix == 0 {
        cards.push(string_card(
            "ROWORDER",
            "BOTTOM-UP",
            "Order of the rows in image array",
        ));
    }
    cards.push(pad("END".into()));

    let mut out = BufWriter::new(File::create(path)?);
    let mut header_bytes: Vec<u8> = cards.concat().into_bytes();
    header_bytes.resize(header_bytes.len().div_ceil(BLOCK) * BLOCK, b' ');
    out.write_all(&header_bytes)?;

    let clip = |v: f64| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let ch = img.channels;
    let mut row = Vec::with_capacity(w * 4);
    let mut written = 0;
    for p in 0..ch {
        for file_row in 0..h {
            let y = if src.top_down {
                file_row
            } else {
                h - 1 - file_row
            };
            row.clear();
            for x in 0..w {
                let v = clip(img.data[(y * w + x) * ch + p]);
                match depth {
                    // Truncates like Siril's float -> ushort conversion.
                    BitDepth::U16 => {
                        let u = (v * 65535.0) as u16;
                        row.extend_from_slice(&((u as i32 - 32768) as i16).to_be_bytes());
                    }
                    _ => row.extend_from_slice(&((v * src.float_scale) as f32).to_be_bytes()),
                }
            }
            out.write_all(&row)?;
            written += row.len();
        }
    }
    out.write_all(&vec![0; (BLOCK - written % BLOCK) % BLOCK])?;
    out.write_all(&src.extensions)?;
    out.flush()?;
    Ok(())
}

fn bad(msg: &str) -> Error {
    Error::Unsupported(format!("invalid FITS file: {msg}"))
}

fn unsupported(msg: &str) -> Error {
    Error::Unsupported(format!("FITS: {msg}"))
}

/// Reads header cards up to (excluding) END.
fn read_cards(r: &mut impl Read) -> Result<Vec<String>, Error> {
    let mut cards = Vec::new();
    let mut block = [0u8; BLOCK];
    loop {
        r.read_exact(&mut block)
            .map_err(|_| bad("header is truncated"))?;
        for chunk in block.as_chunks::<CARD>().0 {
            if !chunk.is_ascii() {
                return Err(bad("header contains non-ASCII characters"));
            }
            let card = String::from_utf8(chunk.to_vec()).expect("ASCII");
            if keyword(&card) == "END" {
                return Ok(cards);
            }
            cards.push(card);
        }
    }
}

fn keyword(card: &str) -> &str {
    card[..8.min(card.len())].trim_end()
}

/// Value text of a `KEY = value / comment` card, without the comment.
fn value_text(card: &str) -> Option<&str> {
    if card.get(8..10) != Some("= ") {
        return None;
    }
    let v = card[10..].trim_start();
    if let Some(body) = v.strip_prefix('\'') {
        // A string ends at the first quote that is not doubled ('').
        let mut i = 0;
        while let Some(j) = body[i..].find('\'') {
            let k = i + j;
            if body[k + 1..].starts_with('\'') {
                i = k + 2;
            } else {
                return Some(&v[..k + 2]);
            }
        }
        return Some(v);
    }
    Some(v.split('/').next().unwrap_or("").trim())
}

fn find<'a>(cards: &'a [String], key: &str) -> Option<&'a str> {
    cards
        .iter()
        .find(|c| keyword(c) == key)
        .and_then(|c| value_text(c))
}

fn unquote(v: &str) -> String {
    v.trim_matches('\'')
        .replace("''", "'")
        .trim_end()
        .to_string()
}

/// Keywords describing the data layout, regenerated on write (or invalid after
/// processing, like checksums and data ranges).
fn is_structural(key: &str) -> bool {
    matches!(
        key,
        "SIMPLE"
            | "BITPIX"
            | "NAXIS"
            | "EXTEND"
            | "BZERO"
            | "BSCALE"
            | "BLANK"
            | "PCOUNT"
            | "GCOUNT"
            | "CHECKSUM"
            | "DATASUM"
            | "DATAMIN"
            | "DATAMAX"
    ) || (key.len() > 5 && key.starts_with("NAXIS") && key[5..].bytes().all(|b| b.is_ascii_digit()))
}

fn pad(mut card: String) -> String {
    card.truncate(CARD);
    format!("{card:<CARD$}")
}

fn with_comment(card: String, comment: &str) -> String {
    pad(if comment.is_empty() {
        card
    } else {
        format!("{card} / {comment}")
    })
}

fn int_card(key: &str, value: i64, comment: &str) -> String {
    with_comment(format!("{key:<8}= {value:>20}"), comment)
}

fn logical_card(key: &str, value: bool, comment: &str) -> String {
    let v = if value { "T" } else { "F" };
    with_comment(format!("{key:<8}= {v:>20}"), comment)
}

fn string_card(key: &str, value: &str, comment: &str) -> String {
    let quoted = format!("'{:<8}'", value.replace('\'', "''"));
    with_comment(format!("{key:<8}= {quoted:<20}"), comment)
}

/// Current UTC time as `YYYY-MM-DDThh:mm:ss`.
fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_values() {
        let s = string_card("OBJECT", "NGC 2244", "Name");
        assert_eq!(s.len(), 80);
        assert_eq!(value_text(&s), Some("'NGC 2244'"));
        let s = pad("NOTE    = 'it''s / here' / comment".into());
        assert_eq!(unquote(value_text(&s).unwrap()), "it's / here");
        let s = int_card("NAXIS1", 4440, "length of data axis 1");
        assert_eq!(&s[..30], "NAXIS1  =                 4440");
        assert_eq!(value_text(&s), Some("4440"));
        assert_eq!(value_text(&pad("HISTORY Crop (x=1)".into())), None);
    }

    #[test]
    fn structural_keywords() {
        for k in ["SIMPLE", "NAXIS", "NAXIS1", "NAXIS3", "BZERO", "CHECKSUM"] {
            assert!(is_structural(k), "{k}");
        }
        for k in ["NAXISX", "OBJECT", "ROWORDER", "HISTORY", "CRPIX1", "DATE"] {
            assert!(!is_structural(k), "{k}");
        }
    }

    #[test]
    fn utc_now_format() {
        let s = utc_now();
        assert_eq!(s.len(), 19);
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[10..11], "T");
    }
}
