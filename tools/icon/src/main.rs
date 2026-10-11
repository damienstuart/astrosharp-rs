//! Renders the app icon SVG into a macOS `.iconset` directory (every size rendered
//! from the vector source, so small sizes stay crisp). Run via `make icon`.
//!
//! Usage: render-icon <icon.svg> <out.iconset>

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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, svg, out] = &args[..] else {
        return Err("usage: render-icon <icon.svg> <out.iconset>".into());
    };
    let tree = usvg::Tree::from_data(&std::fs::read(svg)?, &usvg::Options::default())?;
    let out = Path::new(out);
    std::fs::create_dir_all(out)?;
    for (points, scale) in SIZES {
        let px = points * scale;
        let mut pixmap = tiny_skia::Pixmap::new(px, px).ok_or("invalid icon size")?;
        let s = px as f32 / tree.size().width();
        resvg::render(&tree, tiny_skia::Transform::from_scale(s, s), &mut pixmap.as_mut());
        let suffix = if scale == 2 { "@2x" } else { "" };
        pixmap.save_png(out.join(format!("icon_{points}x{points}{suffix}.png")))?;
    }
    Ok(())
}
