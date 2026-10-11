use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant};

use eframe::egui::{self, ColorImage, Pos2, pos2};

use astrosharp_core::pipeline::{MIN_CHUNK_SIZE, MIN_SIZE};
use astrosharp_core::weights::{PSF_MAX, PSF_MIN, PSF_STEP};
use astrosharp_core::{
    BitDepth, Image, Mode, ModelKind, Params, Progress, process, read_tiff, write_tiff,
};

use crate::viewer::{self, Layers, Picture, Roi, View};

/// Extra context processed around the ROI so the preview matches full-image processing
/// (two stacked 9x9 windows for Hybrid).
const PREVIEW_MARGIN: usize = 8;
/// Delay after the last parameter change before the preview is recomputed.
const PREVIEW_DEBOUNCE: Duration = Duration::from_millis(250);

struct Loaded {
    path: PathBuf,
    image: Arc<Image>,
    picture: Picture,
}

struct Processed {
    params: Params,
    image: Arc<Image>,
    picture: Picture,
}

struct PreviewResult {
    roi: Roi,
    picture: Picture,
}

type JobOutput = Result<(Image, ColorImage, usize), String>;

struct Job {
    params: Params,
    roi: Option<Roi>,
    progress: Arc<Progress>,
    rx: Receiver<JobOutput>,
    started: Instant,
}

impl Job {
    fn spawn(ctx: &egui::Context, image: Arc<Image>, params: Params, roi: Option<Roi>) -> Job {
        let (tx, rx) = mpsc::channel();
        let progress = Arc::new(Progress::new());
        let (ctx, p, pr) = (ctx.clone(), params.clone(), progress.clone());
        let max_side = ctx.input(|i| i.max_texture_side);
        std::thread::spawn(move || {
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
            let msg = result.map_err(|e| e.to_string()).map(|img| {
                let (ci, step) = Picture::color_image(&img, max_side);
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
    split: f32,
    float_output: bool,
    preview: Option<PreviewResult>,
    preview_job: Option<Job>,
    preview_requested: Option<Instant>,
    full: Option<Processed>,
    full_job: Option<Job>,
    last_params: Params,
    status: String,
}

impl AstroSharpApp {
    pub fn new(cc: &eframe::CreationContext<'_>, initial: Option<PathBuf>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        let params = Params::default();
        let mut app = AstroSharpApp {
            last_params: params.clone(),
            params,
            loaded: None,
            roi: None,
            view: View::default(),
            split: 0.5,
            float_output: false,
            preview: None,
            preview_job: None,
            preview_requested: None,
            full: None,
            full_job: None,
            status: "Open a TIFF (or drop one on the window) to begin.".into(),
        };
        if let Some(path) = initial {
            app.open(&cc.egui_ctx, &path);
        }
        app
    }

    fn open(&mut self, ctx: &egui::Context, path: &Path) {
        match read_tiff(path) {
            Ok(img) if img.width < MIN_SIZE || img.height < MIN_SIZE => {
                self.status = format!("Image too small (minimum {MIN_SIZE}x{MIN_SIZE}).");
            }
            Ok(img) => {
                let max_side = ctx.input(|i| i.max_texture_side);
                let (ci, step) = Picture::color_image(&img, max_side);
                let picture = Picture::new(ctx, "original", ci, step, img.width, img.height);
                self.roi = Some(Roi::default_for(img.width, img.height));
                self.status = format!(
                    "{} — {}x{} {}",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    img.width,
                    img.height,
                    if img.is_color() { "RGB" } else { "grey" }
                );
                self.loaded = Some(Loaded {
                    path: path.to_path_buf(),
                    image: Arc::new(img),
                    picture,
                });
                self.preview = None;
                self.preview_job = None;
                self.full = None;
                self.full_job = None;
                self.view.fit();
                self.request_preview();
            }
            Err(e) => self.status = format!("Could not open {}: {e}", path.display()),
        }
    }

    fn request_preview(&mut self) {
        self.preview_requested = Some(Instant::now());
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        if let Some(out) = self.preview_job.as_ref().and_then(Job::poll) {
            let job = self.preview_job.take().unwrap();
            match out {
                Ok((_, ci, step)) => {
                    let roi = job.roi.expect("preview jobs have a ROI");
                    let picture = Picture::new(ctx, "preview", ci, step, roi.w, roi.h);
                    self.preview = Some(PreviewResult { roi, picture });
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
                        format!("Processed in {:.1}s.", job.started.elapsed().as_secs_f32());
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

        if let (Some(t), Some(loaded), Some(roi)) = (self.preview_requested, &self.loaded, self.roi)
        {
            let wait = PREVIEW_DEBOUNCE.saturating_sub(t.elapsed());
            if wait.is_zero() {
                self.preview_requested = None;
                // Replacing a running job cancels it (see `Drop for Job`).
                self.preview_job = Some(Job::spawn(
                    ctx,
                    loaded.image.clone(),
                    self.params.clone(),
                    Some(roi),
                ));
            } else {
                ctx.request_repaint_after(wait);
            }
        }
        if self.preview_job.is_some() || self.full_job.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        ui.heading("AstroSharp");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("Open TIFF…").clicked()
                && let Some(path) = rfd::FileDialog::new()
                    .add_filter("TIFF", &["tif", "tiff"])
                    .pick_file()
            {
                self.open(ui.ctx(), &path);
            }
        });
        ui.separator();

        let p = &mut self.params;
        egui::ComboBox::from_label("Model")
            .selected_text(model_name(p.model))
            .show_ui(ui, |ui| {
                for m in [ModelKind::Psf, ModelKind::AstroClean, ModelKind::Hybrid] {
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
        ui.separator();

        ui.label("View");
        ui.add(
            egui::Slider::new(&mut self.split, 0.0..=1.0)
                .show_value(false)
                .text("Before | after"),
        );
        ui.small("Drag to pan, scroll to zoom, double-click to fit.\nShift-drag or right-drag to choose the preview region.\nHold Space to see the original.");
        ui.separator();

        let loaded = self.loaded.is_some();
        if let Some(job) = &self.full_job {
            ui.add(egui::ProgressBar::new(job.progress.fraction()).show_percentage());
            if ui.button("Cancel").clicked() {
                self.full_job = None;
                self.status = "Processing cancelled.".into();
            }
        } else if ui
            .add_enabled(loaded, egui::Button::new("Process full image"))
            .clicked()
            && let Some(l) = &self.loaded
        {
            self.full_job = Some(Job::spawn(
                ui.ctx(),
                l.image.clone(),
                self.params.clone(),
                None,
            ));
            self.status = "Processing…".into();
        }

        let current = self.full.as_ref().filter(|f| f.params == self.params);
        ui.add_space(4.0);
        ui.checkbox(&mut self.float_output, "Save as 32-bit float");
        if ui
            .add_enabled(current.is_some(), egui::Button::new("Save processed TIFF…"))
            .on_disabled_hover_text("Process the full image with the current settings first.")
            .clicked()
            && let (Some(full), Some(l)) = (current, &self.loaded)
        {
            let stem = l.path.file_stem().unwrap_or_default().to_string_lossy();
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("TIFF", &["tif", "tiff"])
                .set_file_name(format!("{stem}_sharpened.tif"))
                .save_file()
            {
                let depth = if self.float_output {
                    BitDepth::F32
                } else {
                    BitDepth::U16
                };
                self.status = match write_tiff(&path, &full.image, depth) {
                    Ok(()) => format!("Saved {}", path.display()),
                    Err(e) => format!("Save failed: {e}"),
                };
            }
        }

        ui.separator();
        if self.preview_job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Updating preview…");
            });
        }
        ui.label(&self.status);
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
            self.request_preview();
        }

        egui::CentralPanel::no_frame().show(ui, |ui| {
            let Some(loaded) = &self.loaded else {
                ui.centered_and_justified(|ui| ui.label("Drop a TIFF here"));
                return;
            };
            let processed = match (&self.full, &self.preview) {
                (Some(f), _) if f.params == self.params => Some((&f.picture, Pos2::ZERO)),
                (_, Some(p)) => Some((&p.picture, pos2(p.roi.x as f32, p.roi.y as f32))),
                _ => None,
            };
            let layers = Layers {
                original: &loaded.picture,
                processed,
                roi: self.roi,
                split: self.split,
            };
            if let Some(roi) = viewer::show(ui, &mut self.view, layers) {
                self.roi = Some(roi);
                self.request_preview();
            }
        });
    }
}
