//! Zoomable/pannable image view with a draggable before/after divider and ROI selection.

use eframe::egui::{
    self, Color32, ColorImage, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, Vec2, pos2,
    vec2,
};

use astrosharp_core::wcs::{format_dec, format_ra};
use astrosharp_core::{Image, Wcs};

/// A region of interest in image pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Roi {
    pub const MIN_SIDE: usize = 16;
    pub const MAX_SIDE: usize = 2048;

    /// Default preview region: the central 10% of each side (as in the R app),
    /// but at least 256 px.
    pub fn default_for(width: usize, height: usize) -> Roi {
        let side = |n: usize| (n / 10).max(256).min(n).min(Self::MAX_SIDE);
        let (w, h) = (side(width), side(height));
        Roi {
            x: (width - w) / 2,
            y: (height - h) / 2,
            w,
            h,
        }
    }

    fn from_corners(a: Pos2, b: Pos2, width: usize, height: usize) -> Roi {
        let clamp = |v: f32, n: usize| v.round().clamp(0.0, n as f32) as usize;
        let (x0, x1) = (clamp(a.x.min(b.x), width), clamp(a.x.max(b.x), width));
        let (y0, y1) = (clamp(a.y.min(b.y), height), clamp(a.y.max(b.y), height));
        let fit = |lo: usize, hi: usize, n: usize| {
            let len = (hi - lo).clamp(Self::MIN_SIDE.min(n), Self::MAX_SIDE.min(n));
            let lo = lo.min(n - len);
            (lo, len)
        };
        let (x, w) = fit(x0, x1, width);
        let (y, h) = fit(y0, y1, height);
        Roi { x, y, w, h }
    }

    fn rect(&self) -> Rect {
        Rect::from_min_size(
            pos2(self.x as f32, self.y as f32),
            vec2(self.w as f32, self.h as f32),
        )
    }
}

/// A GPU texture showing an image, possibly downsampled to fit the texture size limit.
pub struct Picture {
    pub texture: TextureHandle,
    /// Image pixels per texture pixel.
    step: usize,
    width: usize,
    height: usize,
}

impl Picture {
    /// Converts to 8-bit and downsamples (nearest) so that no side exceeds `max_side`.
    pub fn color_image(img: &Image, max_side: usize) -> (ColorImage, usize) {
        let step = img.width.max(img.height).div_ceil(max_side).max(1);
        let (tw, th) = (img.width.div_ceil(step), img.height.div_ceil(step));
        let q = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        let mut pixels = Vec::with_capacity(tw * th);
        for ty in 0..th {
            for tx in 0..tw {
                let i = (ty * step * img.width + tx * step) * img.channels;
                pixels.push(if img.is_color() {
                    Color32::from_rgb(q(img.data[i]), q(img.data[i + 1]), q(img.data[i + 2]))
                } else {
                    Color32::from_gray(q(img.data[i]))
                });
            }
        }
        (ColorImage::new([tw, th], pixels), step)
    }

    pub fn new(
        ctx: &egui::Context,
        name: &str,
        image: ColorImage,
        step: usize,
        width: usize,
        height: usize,
    ) -> Self {
        let options = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            ..egui::TextureOptions::LINEAR
        };
        Picture {
            texture: ctx.load_texture(name, image, options),
            step,
            width,
            height,
        }
    }

    /// Paints the picture so that its pixel (0, 0) lands at `origin` (image coordinates).
    fn paint(&self, painter: &egui::Painter, view: &View, screen: Rect, origin: Pos2) {
        let size = self.texture.size();
        let covered = vec2((size[0] * self.step) as f32, (size[1] * self.step) as f32);
        let uv = Rect::from_min_max(
            Pos2::ZERO,
            pos2(
                self.width as f32 / covered.x,
                self.height as f32 / covered.y,
            ),
        );
        let img_rect = Rect::from_min_size(origin, vec2(self.width as f32, self.height as f32));
        let rect = Rect::from_min_max(
            view.to_screen(screen, img_rect.min),
            view.to_screen(screen, img_rect.max),
        );
        painter.image(self.texture.id(), rect, uv, Color32::WHITE);
    }
}

/// Zoom/pan state: `zoom` is screen points per image pixel, `center` the image point at
/// the centre of the view.
pub struct View {
    zoom: f32,
    center: Pos2,
    fit_pending: bool,
    drag_anchor: Option<Pos2>,
    /// Live ROI while the user is dragging one out.
    pub selecting: Option<Roi>,
}

impl Default for View {
    fn default() -> Self {
        View {
            zoom: 1.0,
            center: Pos2::ZERO,
            fit_pending: true,
            drag_anchor: None,
            selecting: None,
        }
    }
}

impl View {
    pub fn fit(&mut self) {
        self.fit_pending = true;
    }

    fn to_screen(&self, screen: Rect, p: Pos2) -> Pos2 {
        screen.center() + (p - self.center) * self.zoom
    }

    fn to_image(&self, screen: Rect, p: Pos2) -> Pos2 {
        self.center + (p - screen.center()) / self.zoom
    }
}

pub struct Layers<'a> {
    pub original: &'a Picture,
    /// Processed full image, or processed ROI with its position.
    pub processed: Option<(&'a Picture, Pos2)>,
    /// ROI to outline. When set, shift-drag / right-drag selects a new one.
    pub roi: Option<Roi>,
    /// Before/after divider position as a fraction of the view width (original on the
    /// left), or `None` to show only the processed image. Dragging the divider updates it.
    pub split: Option<&'a mut f32>,
    /// Plate solution, to show the sky coordinates under the cursor.
    pub sky: Option<&'a Wcs>,
}

/// Half-width of the divider's grab area, and the radius of its handle.
const HANDLE_RADIUS: f32 = 14.0;
const ACCENT: Color32 = Color32::from_rgb(255, 200, 40);

/// Shows the image view; returns a new ROI if the user finished selecting one.
pub fn show(ui: &mut egui::Ui, view: &mut View, layers: Layers<'_>) -> Option<Roi> {
    let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
    let screen = response.rect;
    let (w, h) = (layers.original.width, layers.original.height);
    if view.fit_pending {
        view.zoom = (screen.width() / w as f32).min(screen.height() / h as f32);
        view.center = pos2(w as f32 / 2.0, h as f32 / 2.0);
        view.fit_pending = false;
    }

    // Zoom around the cursor with the scroll wheel / pinch.
    if let Some(cursor) = response.hover_pos() {
        let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
        let factor = (scroll * 0.002).exp() * pinch;
        if factor != 1.0 {
            let anchor = view.to_image(screen, cursor);
            view.zoom = (view.zoom * factor).clamp(0.01, 64.0);
            view.center = anchor - (cursor - screen.center()) / view.zoom;
        }
    }
    if response.double_clicked() {
        view.fit();
    }

    // Shift-drag or right-drag selects a region (when a ROI is in use); plain drag pans.
    let selecting = layers.roi.is_some()
        && (ui.input(|i| i.modifiers.shift) || response.dragged_by(egui::PointerButton::Secondary));
    let mut finished = None;
    if response.drag_started() && selecting {
        view.drag_anchor = response
            .interact_pointer_pos()
            .map(|p| view.to_image(screen, p));
    }
    if let (Some(anchor), Some(p)) = (view.drag_anchor, response.interact_pointer_pos()) {
        view.selecting = Some(Roi::from_corners(anchor, view.to_image(screen, p), w, h));
    } else if response.dragged() {
        view.center -= response.drag_delta() / view.zoom;
    }
    if response.drag_stopped() && view.drag_anchor.is_some() {
        view.drag_anchor = None;
        finished = view.selecting.take();
    }

    // The divider sits on top of the image, so it takes drags before the pan/select above.
    let divider_x = layers.split.map(|split| {
        let x = screen.left() + *split * screen.width();
        let grab = Rect::from_x_y_ranges(x - HANDLE_RADIUS..=x + HANDLE_RADIUS, screen.y_range());
        let handle = ui
            .interact(grab, response.id.with("divider"), Sense::drag())
            .on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
        if handle.dragged()
            && let Some(p) = handle.interact_pointer_pos()
        {
            *split = ((p.x - screen.left()) / screen.width()).clamp(0.0, 1.0);
        }
        screen.left() + *split * screen.width()
    });

    painter.rect_filled(screen, 0.0, Color32::from_gray(24));
    layers.original.paint(&painter, view, screen, Pos2::ZERO);

    let show_processed = !ui.input(|i| i.key_down(egui::Key::Space));
    if let (Some((picture, origin)), true) = (layers.processed, show_processed) {
        let split_x = divider_x.unwrap_or(screen.left());
        let clip = Rect::from_min_max(pos2(split_x, screen.top()), screen.max);
        picture.paint(
            &painter.with_clip_rect(clip.intersect(screen)),
            view,
            screen,
            origin,
        );
        if let Some(x) = divider_x {
            paint_divider(&painter, screen, x);
        }
    }

    if let Some(roi) = view.selecting.or(layers.roi) {
        let r = roi.rect();
        let rect = Rect::from_min_max(view.to_screen(screen, r.min), view.to_screen(screen, r.max));
        painter.rect_stroke(rect, 0.0, Stroke::new(1.5, ACCENT), StrokeKind::Outside);
    }

    if let Some(p) = response.hover_pos() {
        let ip = view.to_image(screen, p);
        if ip.x >= 0.0 && ip.y >= 0.0 && (ip.x as usize) < w && (ip.y as usize) < h {
            let mut text = String::new();
            if let Some(wcs) = layers.sky {
                let (ra, dec) = wcs.sky_at(ip.x as f64, ip.y as f64);
                text = format!("α {}  δ {}\n", format_ra(ra), format_dec(dec));
            }
            text += &format!(
                "x {} y {}  |  {:.0}%",
                ip.x as usize,
                ip.y as usize,
                view.zoom * 100.0
            );
            painter.text(
                screen.left_bottom() + Vec2::new(8.0, -8.0),
                egui::Align2::LEFT_BOTTOM,
                text,
                egui::FontId::monospace(12.0),
                Color32::from_white_alpha(180),
            );
        }
    }
    finished
}

/// The before/after line with "original" / "processed" labels and a drag handle.
fn paint_divider(painter: &egui::Painter, screen: Rect, x: f32) {
    painter.vline(
        x,
        screen.y_range(),
        Stroke::new(1.5, Color32::from_white_alpha(180)),
    );
    let label = |pos: Pos2, align, text: &str| {
        painter.text(
            pos,
            align,
            text,
            egui::FontId::proportional(13.0),
            Color32::from_white_alpha(200),
        );
    };
    label(
        pos2(x - 8.0, screen.top() + 8.0),
        egui::Align2::RIGHT_TOP,
        "original",
    );
    label(
        pos2(x + 8.0, screen.top() + 8.0),
        egui::Align2::LEFT_TOP,
        "processed",
    );

    let c = pos2(x, screen.center().y);
    painter.circle(
        c,
        HANDLE_RADIUS,
        ACCENT,
        Stroke::new(1.5, Color32::from_black_alpha(140)),
    );
    let dark = Color32::from_rgb(43, 34, 0);
    for dir in [-1.0, 1.0] {
        let tip = c + vec2(dir * 9.0, 0.0);
        let base = c + vec2(dir * 3.0, 0.0);
        painter.add(egui::Shape::convex_polygon(
            // Clockwise winding (egui anti-aliases convex polygons along the outside).
            vec![
                tip,
                base + vec2(0.0, 5.5 * dir),
                base + vec2(0.0, -5.5 * dir),
            ],
            dark,
            Stroke::NONE,
        ));
    }
}
