//! The histogram panel, as in the R app: the result's histogram, with the star mask's
//! beta curve (Blend, Blend 2) drawn over it.

use astrosharp_core::pbeta::PBeta;
use eframe::egui::{self, Color32, Pos2, Stroke, pos2, vec2};

/// Bins over [0, 1], as the R app asks for.
pub const BINS: usize = 1000;
const HEIGHT: f32 = 100.0;
const BACKGROUND: Color32 = Color32::from_rgb(0x25, 0x25, 0x25);
const BARS: Color32 = Color32::from_gray(190);
const CURVE: Color32 = Color32::from_rgb(235, 64, 52);

/// Draws `bins` (counts normalised to a maximum of 1) and, if given, the CDF of the beta
/// distribution with these shape parameters.
pub fn show(ui: &mut egui::Ui, bins: Option<&[f32]>, curve: Option<(f64, f64)>) {
    let (rect, _) =
        ui.allocate_exact_size(vec2(ui.available_width(), HEIGHT), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BACKGROUND);
    let inner = rect.shrink(3.0);
    let columns = inner.width().floor().max(1.0) as usize;
    if let Some(bins) = bins.filter(|b| !b.is_empty()) {
        // Each screen column shows the fullest of the bins it covers.
        for x in 0..columns {
            let lo = x * bins.len() / columns;
            let hi = ((x + 1) * bins.len() / columns).clamp(lo + 1, bins.len());
            let v = bins[lo..hi].iter().copied().fold(0.0, f32::max);
            if v > 0.0 {
                let px = inner.left() + x as f32 + 0.5;
                let top = inner.bottom() - v * inner.height();
                painter.line_segment(
                    [pos2(px, inner.bottom()), pos2(px, top)],
                    Stroke::new(1.0, BARS),
                );
            }
        }
    }
    if let Some((a, b)) = curve {
        let pb = PBeta::new(a, b);
        let points: Vec<Pos2> = (0..=columns)
            .map(|i| {
                let x = i as f64 / columns as f64;
                pos2(
                    inner.left() + i as f32,
                    inner.bottom() - pb.at(x) as f32 * inner.height(),
                )
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(2.0, CURVE)));
    }
}
