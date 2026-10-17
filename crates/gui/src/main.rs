mod app;
mod viewer;

fn main() -> eframe::Result {
    let telemetry = astrosharp_telemetry::init("astrosharp-gui");
    // Optional image to open at startup: `astrosharp-gui image.tif`.
    let initial = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    // Window/Dock icon when not running from the app bundle (`make icon` regenerates it).
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/icon-512.png"))
        .expect("embedded icon is a valid PNG");
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("AstroSharp")
            .with_icon(icon)
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "AstroSharp",
        options,
        Box::new(|cc| Ok(Box::new(app::AstroSharpApp::new(cc, initial, telemetry)))),
    )
}
