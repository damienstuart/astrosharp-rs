//! Renders the app icon SVG into a macOS `.iconset` directory and, optionally, a Windows
//! `.ico` file (every size rendered from the vector source, so small sizes stay crisp).
//! Run via `make icon`.
//!
//! Usage: render-icon <icon.svg> <out.iconset> [<out.ico>]

use std::error::Error;
use std::path::Path;

use resvg::{tiny_skia, usvg};

/// (point size, scale) pairs required by `iconutil`.
const SIZES: [(u32, u32); 10] = [
    (16, 1),
    (16, 2),
    (32, 1),
    (32, 2),
    (128, 1),
    (128, 2),
    (256, 1),
    (256, 2),
    (512, 1),
    (512, 2),
];

/// Pixel sizes Windows picks from for title bars, Explorer views, the taskbar and the
/// Start menu at the usual display scalings.
const ICO_SIZES: [u32; 9] = [16, 20, 24, 32, 40, 48, 64, 128, 256];

fn render(tree: &usvg::Tree, px: u32) -> Result<tiny_skia::Pixmap, Box<dyn Error>> {
    let mut pixmap = tiny_skia::Pixmap::new(px, px).ok_or("invalid icon size")?;
    let s = px as f32 / tree.size().width();
    resvg::render(tree, tiny_skia::Transform::from_scale(s, s), &mut pixmap.as_mut());
    Ok(pixmap)
}

/// Writes an ICO file holding one PNG-compressed 32-bit image per size.
fn write_ico(tree: &usvg::Tree, path: &Path) -> Result<(), Box<dyn Error>> {
    let images = ICO_SIZES
        .iter()
        .map(|&px| Ok((px, render(tree, px)?.encode_png()?)))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let count = images.len() as u16;
    // ICONDIR: reserved, type 1 (icon), image count.
    let mut ico = [0u16, 1, count]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>();
    let mut offset = 6 + 16 * images.len() as u32;
    for (px, png) in &images {
        // ICONDIRENTRY: width and height (0 means 256), no palette, reserved, one colour
        // plane, 32 bits per pixel, data size and offset.
        let side = if *px >= 256 { 0 } else { *px as u8 };
        ico.extend_from_slice(&[side, side, 0, 0]);
        ico.extend_from_slice(&1u16.to_le_bytes());
        ico.extend_from_slice(&32u16.to_le_bytes());
        ico.extend_from_slice(&(png.len() as u32).to_le_bytes());
        ico.extend_from_slice(&offset.to_le_bytes());
        offset += png.len() as u32;
    }
    for (_, png) in &images {
        ico.extend_from_slice(png);
    }
    std::fs::write(path, ico)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (svg, out, ico) = match &args[..] {
        [_, svg, out] => (svg, out, None),
        [_, svg, out, ico] => (svg, out, Some(ico)),
        _ => return Err("usage: render-icon <icon.svg> <out.iconset> [<out.ico>]".into()),
    };
    let tree = usvg::Tree::from_data(&std::fs::read(svg)?, &usvg::Options::default())?;
    let out = Path::new(out);
    std::fs::create_dir_all(out)?;
    for (points, scale) in SIZES {
        let suffix = if scale == 2 { "@2x" } else { "" };
        render(&tree, points * scale)?
            .save_png(out.join(format!("icon_{points}x{points}{suffix}.png")))?;
    }
    if let Some(ico) = ico {
        write_ico(&tree, Path::new(ico))?;
    }
    Ok(())
}
