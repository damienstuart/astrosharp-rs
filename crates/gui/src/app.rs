use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, ColorImage, Pos2, pos2};
use tracing::field::Empty;
use tracing::info_span;

use astrosharp_core::pipeline::{MIN_CHUNK_SIZE, MIN_SIZE};
use astrosharp_core::weights::{PSF_MAX, PSF_MIN, PSF_STEP};
use astrosharp_core::{
    BitDepth, Error, FileFormat, FitsHeader, Image, Mode, ModelKind, Params, Progress, SaveOptions,
    VERSION, Wcs, process, read_image, save_image,
};
use astrosharp_telemetry::{Telemetry, record_error};

use crate::viewer::{self, Layers, Picture, Roi, View};

/// Extra context processed around the ROI so the preview matches full-image processing
/// (two stacked 9x9 windows for Hybrid).
const PREVIEW_MARGIN: usize = 8;
/// Delay after the last change before the image (or preview region) is reprocessed.
const UPDATE_DEBOUNCE: Duration = Duration::from_millis(250);

// Button colours by kind of operation. Each keeps its text at WCAG AA contrast (4.5:1) when
// idle, hovered and pressed, and stands out from the dark panel by at least 3:1.
/// Processing.
const PROCESS_FILL: Color32 = Color32::from_rgb(0x28, 0x62, 0xc8);
/// Discards the result but keeps the settings.
const RESET_IMAGE_FILL: Color32 = Color32::from_rgb(0xd0, 0x8c, 0x0c);
/// Discards the result and the settings.
const RESET_ALL_FILL: Color32 = Color32::from_rgb(0xc0, 0x39, 0x2b);
const LIGHT_TEXT: Color32 = Color32::WHITE;
const DARK_TEXT: Color32 = Color32::from_gray(20);

/// What a reset restores besides reloading the image.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reset {
    /// Nothing: the settings, view and preview region are kept.
    Image,
    /// The default settings, and a fresh view and preview region.
    All,
}

struct Loaded {
    path: PathBuf,
    image: Arc<Image>,
    /// Header of a FITS source, preserved when saving as FITS.
    fits: Option<FitsHeader>,
    /// Plate solution from the FITS header, for the cursor's sky coordinates.
    wcs: Option<Wcs>,
    /// The image looks unstretched (see [`Image::looks_linear`]).
    linear: bool,
    picture: Picture,
}

impl Loaded {
    /// Reads `path` and prepares it for display.
    fn read(ctx: &egui::Context, path: &Path) -> Result<Loaded, String> {
        let span = info_span!(
            "open_image",
            path = %path.display(),
            linear = Empty,
            otel.status_description = Empty,
        );
        let result = span.in_scope(|| {
            let (img, fits) = read_image(path).map_err(|e| e.to_string())?;
            if img.width < MIN_SIZE || img.height < MIN_SIZE {
                return Err(format!("image too small (minimum {MIN_SIZE}x{MIN_SIZE})"));
            }
            let linear = img.looks_linear();
            span.record("linear", linear);
            let max_side = ctx.input(|i| i.max_texture_side);
            let picture = info_span!("prepare_display").in_scope(|| {
                let (ci, step) = Picture::color_image(&img, max_side);
                Picture::new(ctx, "original", ci, step, img.width, img.height)
            });
            Ok(Loaded {
                path: path.to_path_buf(),
                linear,
                picture,
                wcs: fits.as_ref().and_then(|h| Wcs::from_header(h, img.height)),
                image: Arc::new(img),
                fits,
            })
        });
        if let Err(e) = &result {
            record_error(&span, e);
        }
        result
    }

    /// File name, size and type, for the status line.
    fn summary(&self) -> String {
        let img = &self.image;
        let mut s = format!(
            "{} — {}x{} {}",
            self.path.file_name().unwrap_or_default().to_string_lossy(),
            img.width,
            img.height,
            if img.is_color() { "RGB" } else { "grey" }
        );
        if let Some(depth) = self.fits.as_ref().and_then(FitsHeader::source_depth) {
            let solved = if self.wcs.is_some() {
                ", plate-solved"
            } else {
                ""
            };
            s += &format!(" (FITS, {}{solved})", depth.name());
        }
        s
    }
}

struct Processed {
    params: Params,
    image: Arc<Image>,
    picture: Picture,
}

struct PreviewResult {
    params: Params,
    roi: Roi,
    picture: Picture,
}

type JobOutput = Result<(Image, ColorImage, usize), String>;

/// A save running in the background.
struct SaveJob {
    path: PathBuf,
    rx: Receiver<Result<(), String>>,
}

struct Job {
    params: Params,
    roi: Option<Roi>,
    progress: Arc<Progress>,
    rx: Receiver<JobOutput>,
    started: Instant,
}

impl Job {
    fn spawn(ctx: &egui::Context, loaded: &Loaded, params: Params, roi: Option<Roi>) -> Job {
        let (tx, rx) = mpsc::channel();
        let progress = Arc::new(Progress::new());
        let (ctx, p, pr) = (ctx.clone(), params.clone(), progress.clone());
        let image = loaded.image.clone();
        let max_side = ctx.input(|i| i.max_texture_side);
        let span = info_span!(
            "process_job",
            kind = if roi.is_some() { "preview" } else { "full" },
            path = %loaded.path.display(),
            params = %params.describe(),
            roi = roi.map(|r| format!("{}x{} at {},{}", r.w, r.h, r.x, r.y)),
            outcome = Empty,
            otel.status_description = Empty,
        );
        std::thread::spawn(move || {
            let _enter = span.enter();
            let result = match roi {
                None => process(&image, &p, &pr),
                Some(roi) => {
                    let x0 = roi.x.saturating_sub(PREVIEW_MARGIN);
                    let y0 = roi.y.saturating_sub(PREVIEW_MARGIN);
                    let x1 = (roi.x + roi.w + PREVIEW_MARGIN).min(image.width);
                    let y1 = (roi.y + roi.h + PREVIEW_MARGIN).min(image.height);
                    process(&image.crop(y0..y1, x0..x1), &p, &pr).map(|out| {
                        out.crop(
                            roi.y - y0..roi.y - y0 + roi.h,
                            roi.x - x0..roi.x - x0 + roi.w,
                        )
                    })
                }
            };
            let outcome = match &result {
                Ok(_) => "ok",
                Err(Error::Cancelled) => "cancelled",
                Err(e) => {
                    record_error(&span, e);
                    "failed"
                }
            };
            span.record("outcome", outcome);
            let msg = result.map_err(|e| e.to_string()).map(|img| {
                let (ci, step) =
                    info_span!("prepare_display").in_scope(|| Picture::color_image(&img, max_side));
                (img, ci, step)
            });
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
        Job {
            params,
            roi,
            progress,
            rx,
            started: Instant::now(),
        }
    }

    fn poll(&self) -> Option<JobOutput> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("worker stopped unexpectedly".into())),
        }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.progress.cancel();
    }
}

pub struct AstroSharpApp {
    params: Params,
    loaded: Option<Loaded>,
    roi: Option<Roi>,
    view: View,
    /// Show a before/after split.
    compare: bool,
    split: f32,
    /// Process only the preview region automatically; the full image on request.
    preview_only: bool,
    /// Reprocess automatically when settings change.
    live: bool,
    /// Output format for saving, with each format's own options.
    save_format: FileFormat,
    tiff_depth: BitDepth,
    png_depth: BitDepth,
    jpeg_quality: u8,
    fits_depth: BitDepth,
    save_job: Option<SaveJob>,
    preview: Option<PreviewResult>,
    preview_job: Option<Job>,
    update_requested: Option<Instant>,
    full: Option<Processed>,
    full_job: Option<Job>,
    last_params: Params,
    status: String,
    /// Flushed on exit: Cmd-Q on macOS never returns from the event loop to `main`.
    telemetry: Telemetry,
}

impl AstroSharpApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        initial: Option<PathBuf>,
        telemetry: Telemetry,
    ) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let params = default_params();
        let mut app = AstroSharpApp {
            last_params: params.clone(),
            params,
            loaded: None,
            roi: None,
            view: View::default(),
            compare: false,
            split: 0.5,
            preview_only: false,
            live: false,
            save_format: FileFormat::Tiff,
            tiff_depth: FileFormat::Tiff.default_depth(),
            png_depth: FileFormat::Png.default_depth(),
            fits_depth: FileFormat::Fits.default_depth(),
            jpeg_quality: SaveOptions::default().jpeg_quality,
            save_job: None,
            preview: None,
            preview_job: None,
            update_requested: None,
            full: None,
            full_job: None,
            status: "Open a TIFF or FITS file (or drop one on the window) to begin.".into(),
            telemetry,
        };
        if let Some(path) = initial {
            app.open(&cc.egui_ctx, &path);
        }
        app
    }

    fn open(&mut self, ctx: &egui::Context, path: &Path) {
        match Loaded::read(ctx, path) {
            Ok(loaded) => {
                // Default to saving FITS like the source, so its header is kept.
                if let Some(depth) = loaded.fits.as_ref().and_then(FitsHeader::source_depth) {
                    self.save_format = FileFormat::Fits;
                    self.fits_depth = depth;
                }
                self.show_original(loaded, false);
            }
            Err(e) => self.status = format!("Could not open {}: {e}", path.display()),
        }
    }

    /// Discards any results and reloads the open file from disk; [`Reset::All`] also restores
    /// the default settings. If the file can no longer be read, the image as first loaded is
    /// shown instead.
    fn reset(&mut self, ctx: &egui::Context, what: Reset) {
        let Some(path) = self.loaded.as_ref().map(|l| l.path.clone()) else {
            return;
        };
        let kind = match what {
            Reset::Image => "image",
            Reset::All => "all",
        };
        let _span = info_span!("reset", kind, path = %path.display()).entered();
        if what == Reset::All {
            self.params = default_params();
            // Not a settings change: Live update shouldn't reprocess straight away.
            self.last_params = self.params.clone();
        }
        let keep_view = what == Reset::Image;
        match Loaded::read(ctx, &path) {
            Ok(loaded) => self.show_original(loaded, keep_view),
            Err(e) => {
                self.discard_results();
                if !keep_view {
                    self.view.fit();
                }
                self.status = format!(
                    "Could not reload {}: {e}. Showing the image as first loaded.",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
            }
        }
    }

    /// Shows `loaded` as the original, with no results. The view and preview region start
    /// fresh unless `keep_view` is set and the image is the same size as before.
    fn show_original(&mut self, loaded: Loaded, keep_view: bool) {
        let size = |l: &Loaded| (l.image.width, l.image.height);
        if !(keep_view && self.loaded.as_ref().map(size) == Some(size(&loaded))) {
            self.roi = Some(Roi::default_for(loaded.image.width, loaded.image.height));
            self.view.fit();
        }
        self.status = loaded.summary();
        self.loaded = Some(loaded);
        self.discard_results();
    }

    /// Drops the processed results and cancels any processing in progress.
    fn discard_results(&mut self) {
        self.preview = None;
        self.preview_job = None;
        self.full = None;
        self.full_job = None;
        self.update_requested = None;
    }

    /// Reprocesses the image (or the preview region) once changes settle.
    fn request_update(&mut self) {
        self.update_requested = Some(Instant::now());
    }

    /// Whether the displayed result was made with different settings.
    fn result_is_stale(&self) -> bool {
        if self.preview_only {
            self.preview
                .as_ref()
                .is_some_and(|p| p.params != self.params || Some(p.roi) != self.roi)
        } else {
            self.full.is_some() && !self.full_is_current()
        }
    }

    fn full_is_current(&self) -> bool {
        self.full.as_ref().is_some_and(|f| f.params == self.params)
            || self
                .full_job
                .as_ref()
                .is_some_and(|j| j.params == self.params)
    }

    fn start_full(&mut self, ctx: &egui::Context) {
        if let Some(l) = &self.loaded {
            // Replacing a running job cancels it (see `Drop for Job`).
            self.full_job = Some(Job::spawn(ctx, l, self.params.clone(), None));
            self.status = "Processing…".into();
        }
    }

    /// Saves the current full-image result to `path` in the background. A recognised
    /// extension in `path` picks the format; otherwise the selected format's is added.
    fn start_save(&mut self, ctx: &egui::Context, mut path: PathBuf) {
        let Some(full) = self.full.as_ref().filter(|f| f.params == self.params) else {
            return;
        };
        let chosen = self.save_format;
        let format = FileFormat::from_path(&path).unwrap_or_else(|| {
            path.as_mut_os_string()
                .push(format!(".{}", chosen.extensions()[0]));
            chosen
        });
        let opts = SaveOptions {
            format,
            depth: match format {
                FileFormat::Tiff => self.tiff_depth,
                FileFormat::Fits => self.fits_depth,
                FileFormat::Png => self.png_depth,
                FileFormat::Jpeg => BitDepth::U8,
            },
            jpeg_quality: self.jpeg_quality,
        };
        // FITS output keeps the source header (if any) and records the processing.
        let header = (format == FileFormat::Fits).then(|| {
            let mut h = self
                .loaded
                .as_ref()
                .and_then(|l| l.fits.clone())
                .unwrap_or_default();
            h.add_history(&format!("AstroSharp {VERSION}: {}", full.params.describe()));
            h
        });
        let (tx, rx) = mpsc::channel();
        let (image, target, ctx) = (full.image.clone(), path.clone(), ctx.clone());
        let span = info_span!(
            "save_job",
            path = %path.display(),
            otel.status_description = Empty,
        );
        std::thread::spawn(move || {
            let result = span.in_scope(|| save_image(&target, &image, &opts, header.as_ref()));
            if let Err(e) = &result {
                record_error(&span, e);
            }
            let _ = tx.send(result.map_err(|e| e.to_string()));
            ctx.request_repaint();
        });
        self.save_job = Some(SaveJob { path, rx });
        self.status = "Saving…".into();
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        if let Some(out) = self.preview_job.as_ref().and_then(Job::poll) {
            let job = self.preview_job.take().unwrap();
            match out {
                Ok((_, ci, step)) => {
                    let roi = job.roi.expect("preview jobs have a ROI");
                    let picture = Picture::new(ctx, "preview", ci, step, roi.w, roi.h);
                    self.preview = Some(PreviewResult {
                        params: job.params.clone(),
                        roi,
                        picture,
                    });
                }
                Err(e) if e != "cancelled" => self.status = format!("Preview failed: {e}"),
                Err(_) => {}
            }
        }
        if let Some(out) = self.full_job.as_ref().and_then(Job::poll) {
            let job = self.full_job.take().unwrap();
            match out {
                Ok((img, ci, step)) => {
                    let picture = Picture::new(ctx, "processed", ci, step, img.width, img.height);
                    self.status =
                        format!("Processed in {:.2}s.", job.started.elapsed().as_secs_f32());
                    self.full = Some(Processed {
                        params: job.params.clone(),
                        image: Arc::new(img),
                        picture,
                    });
                }
                Err(e) if e == "cancelled" => self.status = "Processing cancelled.".into(),
                Err(e) => self.status = format!("Processing failed: {e}"),
            }
        }

        if let Some(t) = self.update_requested
            && self.loaded.is_some()
        {
            let wait = UPDATE_DEBOUNCE.saturating_sub(t.elapsed());
            if !wait.is_zero() {
                ctx.request_repaint_after(wait);
            } else {
                self.update_requested = None;
                if !self.preview_only {
                    if !self.full_is_current() {
                        self.start_full(ctx);
                    }
                } else if let (Some(l), Some(roi)) = (&self.loaded, self.roi) {
                    if self.full_job.take_if(|j| j.params != self.params).is_some() {
                        self.status = "Full-image processing cancelled (settings changed).".into();
                    }
                    let params = self.params.clone();
                    self.preview_job = Some(Job::spawn(ctx, l, params, Some(roi)));
                }
            }
        }
        if let Some(job) = &self.save_job {
            let result = match job.rx.try_recv() {
                Ok(r) => Some(r),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err("save stopped unexpectedly".into())),
            };
            if let Some(result) = result {
                let job = self.save_job.take().unwrap();
                self.status = match result {
                    Ok(()) => format!(
                        "Saved {}",
                        job.path.file_name().unwrap_or_default().to_string_lossy()
                    ),
                    Err(e) => format!("Save failed: {e}"),
                };
            }
        }
        if self.preview_job.is_some() || self.full_job.is_some() || self.save_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.heading("AstroSharp");
        ui.add_space(6.0);
        if ui.button("Open image…").clicked()
            && let Some(path) = rfd::FileDialog::new()
                .add_filter("TIFF or FITS", &["tif", "tiff", "fit", "fits", "fts"])
                .pick_file()
        {
            self.open(ui.ctx(), &path);
        }
        ui.separator();

        let p = &mut self.params;
        egui::ComboBox::from_label("Model")
            .selected_text(model_name(p.model))
            .show_ui(ui, |ui| {
                for m in [ModelKind::Hybrid, ModelKind::Psf, ModelKind::AstroClean] {
                    ui.selectable_value(&mut p.model, m, model_name(m));
                }
            });
        ui.add_space(4.0);
        ui.add_enabled(
            p.model != ModelKind::AstroClean,
            egui::Slider::new(&mut p.psf_fwhm, PSF_MIN..=PSF_MAX)
                .step_by(PSF_STEP)
                .text("PSF FWHM (px)"),
        )
        .on_hover_text("Measure the FWHM of a small, unsaturated star and divide by 2.35.");
        ui.add(
            egui::Slider::new(&mut p.aggressiveness, 0.01..=1.0)
                .step_by(0.01)
                .text("Aggressiveness"),
        );

        if p.model == ModelKind::Hybrid {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Mode");
                ui.selectable_value(&mut p.mode, Mode::Seamless, "Seamless")
                    .on_hover_text("Whole-image passes, no chunk seams.");
                ui.selectable_value(&mut p.mode, Mode::Parity, "R parity")
                    .on_hover_text(
                        "Reproduces the R app's chunked output exactly (preview is approximate).",
                    );
            });
            if p.mode == Mode::Parity {
                ui.add(
                    egui::Slider::new(&mut p.chunk_size, MIN_CHUNK_SIZE.max(50)..=750)
                        .text("Chunk size"),
                );
            }
        }
        ui.add_space(4.0);
        if ui
            .checkbox(&mut self.live, "Live update")
            .on_hover_text("Reprocess automatically whenever a setting changes.")
            .changed()
            && self.live
        {
            self.request_update();
        }
        ui.separator();

        ui.label("View");
        ui.checkbox(&mut self.compare, "Compare before/after")
            .on_hover_text("Drag the divider on the image to move it.");
        if ui
            .checkbox(&mut self.preview_only, "Preview region only")
            .on_hover_text(
                "Process just a region while adjusting settings; process the full image on request.",
            )
            .changed()
        {
            if !self.preview_only {
                self.preview_job = None;
            }
            if self.live {
                self.request_update();
            }
        }
        let mut help = String::from("Drag to pan, scroll to zoom, double-click to fit.\n");
        if self.compare {
            help += "Drag the divider to compare.\n";
        }
        if self.preview_only {
            help += "Shift-drag or right-drag to choose the region.\n";
        }
        help += "Hold Space to see the original.";
        ui.small(help);
        ui.separator();

        if let Some(job) = &self.full_job {
            ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
            if ui.button("Cancel").clicked() {
                self.full_job = None;
                self.status = "Processing cancelled.".into();
            }
        } else {
            let loaded = self.loaded.is_some();
            ui.horizontal(|ui| {
                let process = |ui: &mut egui::Ui, enabled, label| {
                    tinted_button(ui, enabled, label, PROCESS_FILL, LIGHT_TEXT).clicked()
                };
                if self.preview_only {
                    if process(ui, loaded, "Update preview") {
                        self.request_update();
                    }
                    if process(ui, loaded, "Process full image") {
                        self.start_full(ui.ctx());
                    }
                } else if process(ui, loaded && !self.full_is_current(), "Process image") {
                    self.start_full(ui.ctx());
                }
            });
        }
        if !self.live && self.result_is_stale() {
            ui.small(if self.preview_only {
                "Settings or region changed since the last update."
            } else {
                "Settings changed since the last update."
            });
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let loaded = self.loaded.is_some();
            if tinted_button(ui, loaded, "Reset image", RESET_IMAGE_FILL, DARK_TEXT)
                .on_hover_text(
                    "Discard the processed result and reload the image from disk. \
                     The settings, zoom and preview region are kept.",
                )
                .clicked()
            {
                self.reset(ui.ctx(), Reset::Image);
            }
            if tinted_button(ui, loaded, "Reset all", RESET_ALL_FILL, LIGHT_TEXT)
                .on_hover_text(
                    "Discard the processed result, reload the image from disk and \
                     restore the default settings.",
                )
                .clicked()
            {
                self.reset(ui.ctx(), Reset::All);
            }
        });
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Save as");
            egui::ComboBox::from_id_salt("save-format")
                .selected_text(self.save_format.name())
                .show_ui(ui, |ui| {
                    for f in FileFormat::ALL {
                        ui.selectable_value(&mut self.save_format, f, f.name());
                    }
                });
        });
        match self.save_format {
            FileFormat::Tiff => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tiff_depth, BitDepth::U16, "16-bit")
                        .on_hover_text("Integer samples, like the R app. Opens everywhere.");
                    ui.selectable_value(&mut self.tiff_depth, BitDepth::F32, "32-bit float")
                        .on_hover_text("Full precision, for further processing.");
                });
            }
            FileFormat::Fits => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.fits_depth, BitDepth::F32, "32-bit float")
                        .on_hover_text("Siril's working format.");
                    ui.selectable_value(&mut self.fits_depth, BitDepth::U16, "16-bit")
                        .on_hover_text("Unsigned 16-bit integers.");
                });
            }
            FileFormat::Png => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.png_depth, BitDepth::U16, "16-bit")
                        .on_hover_text("Lossless, full precision.");
                    ui.selectable_value(&mut self.png_depth, BitDepth::U8, "8-bit")
                        .on_hover_text("Lossless; smaller files for sharing.");
                });
            }
            FileFormat::Jpeg => {
                ui.add(egui::Slider::new(&mut self.jpeg_quality, 50..=100).text("Quality"))
                    .on_hover_text("Lossy 8-bit; smallest files.");
            }
        }

        let current = self.full.as_ref().is_some_and(|f| f.params == self.params);
        if ui
            .add_enabled(
                current && self.save_job.is_none(),
                egui::Button::new("Save processed image…"),
            )
            .on_disabled_hover_text(
                "Available once the full image has been processed with the current settings.",
            )
            .clicked()
            && let Some(l) = &self.loaded
        {
            let format = self.save_format;
            let stem = l.path.file_stem().unwrap_or_default().to_string_lossy();
            let picked = rfd::FileDialog::new()
                .add_filter(format.name(), format.extensions())
                .set_file_name(format!("{stem}_sharpened.{}", format.extensions()[0]))
                .save_file();
            if let Some(path) = picked {
                self.start_save(ui.ctx(), path);
            }
        }

        ui.separator();
        if self.preview_job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Updating preview…");
            });
        }
        if self.save_job.is_some() {
            ui.spinner();
        }
        if self.loaded.as_ref().is_some_and(|l| l.linear) {
            ui.label(
                egui::RichText::new(
                    "This image looks linear (unstretched). AstroSharp is meant as a final \
                     touch-up on a stretched image; stretch it first for best results.",
                )
                .color(egui::Color32::from_rgb(255, 200, 40)),
            );
        }
        ui.label(&self.status);

        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.add_space(8.0);
            if ui.button("Quit").clicked() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
    }
}

/// A button filled with `fill` (lighter when hovered, darker when pressed) with `text`
/// coloured text. egui's usual hover, press and disabled behaviour is kept.
fn tinted_button(
    ui: &mut egui::Ui,
    enabled: bool,
    label: &str,
    fill: Color32,
    text: Color32,
) -> egui::Response {
    ui.scope(|ui| {
        let w = &mut ui.visuals_mut().widgets;
        for (state, fill) in [
            (&mut w.inactive, fill),
            (&mut w.hovered, fill.lerp_to_gamma(Color32::WHITE, 0.12)),
            (&mut w.active, fill.lerp_to_gamma(Color32::BLACK, 0.15)),
        ] {
            state.weak_bg_fill = fill;
            state.fg_stroke.color = text;
        }
        ui.add_enabled(enabled, egui::Button::new(label))
    })
    .inner
}

/// The R app's defaults, but with the Hybrid model.
fn default_params() -> Params {
    Params {
        model: ModelKind::Hybrid,
        ..Params::default()
    }
}

fn model_name(m: ModelKind) -> &'static str {
    match m {
        ModelKind::Psf => "PSF (sharpen)",
        ModelKind::AstroClean => "AstroClean (denoise)",
        ModelKind::Hybrid => "Hybrid (sharpen + clean)",
    }
}

impl eframe::App for AstroSharpApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(path) =
            ctx.input(|i| i.raw.dropped_files.first().map(|f| f.path().to_path_buf()))
        {
            self.open(&ctx, &path);
        }
        self.poll_jobs(&ctx);

        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(290.0)
            .show(ui, |ui| self.controls(ui));

        if self.params != self.last_params {
            self.last_params = self.params.clone();
            if self.live {
                self.request_update();
            }
        }

        egui::CentralPanel::no_frame().show(ui, |ui| {
            let Some(loaded) = &self.loaded else {
                ui.centered_and_justified(|ui| ui.label("Drop a TIFF or FITS file here"));
                return;
            };
            // Without the preview region, keep showing the last full result while a new
            // one is computed.
            let full = |f: &&Processed| !self.preview_only || f.params == self.params;
            let processed = match (self.full.as_ref().filter(full), &self.preview) {
                (Some(f), _) => Some((&f.picture, Pos2::ZERO)),
                (None, Some(p)) if self.preview_only => {
                    Some((&p.picture, pos2(p.roi.x as f32, p.roi.y as f32)))
                }
                _ => None,
            };
            let layers = Layers {
                original: &loaded.picture,
                processed,
                roi: self.roi.filter(|_| self.preview_only),
                split: self.compare.then_some(&mut self.split),
                sky: loaded.wcs.as_ref(),
            };
            if let Some(roi) = viewer::show(ui, &mut self.view, layers) {
                self.roi = Some(roi);
                if self.live {
                    self.request_update();
                }
            }
        });
    }

    fn on_exit(&mut self) {
        self.telemetry.shutdown();
    }
}
