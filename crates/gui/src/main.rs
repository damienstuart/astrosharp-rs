// Release builds on Windows run without a console window.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod app;
mod memory;
mod viewer;

fn main() -> eframe::Result {
    let telemetry = astrosharp_telemetry::init("astrosharp-gui");
    // Optional image to open at startup: `astrosharp-gui image.tif`.
    let initial = std::env::args_os().nth(1).map(std::path::PathBuf::from);
    // Window/Dock icon when not running from the app bundle (`make icon` regenerates it).
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../../../assets/icon-512.png"))
        .expect("embedded icon is a valid PNG");
    #[allow(unused_mut)]
    let mut options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("AstroSharp")
            // Lets Linux desktops match windows to the launcher (tools/install_linux.sh).
            .with_app_id("astrosharp")
            .with_icon(icon)
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([800.0, 500.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    #[cfg(target_os = "linux")]
    linux::prefer_x11(&mut options);
    eframe::run_native(
        "AstroSharp",
        options,
        Box::new(|cc| Ok(Box::new(app::AstroSharpApp::new(cc, initial, telemetry)))),
    )
}

#[cfg(target_os = "linux")]
mod linux {
    /// winit only supports dropping files on X11, so on Wayland desktops run through
    /// XWayland when its X server accepts connections. `ASTROSHARP_WAYLAND=1` keeps native
    /// Wayland, which is sharper with fractional scaling but can't receive dropped files.
    pub fn prefer_x11(options: &mut eframe::NativeOptions) {
        if std::env::var_os("ASTROSHARP_WAYLAND").is_some() || std::env::var_os("DISPLAY").is_none()
        {
            return;
        }
        // winit can't fall back once its X11 backend has failed, so check first.
        if !x11_reachable() {
            tracing::warn!(
                "the X server isn't reachable; using Wayland, so dropped files won't open"
            );
            return;
        }
        options.event_loop_builder = Some(Box::new(|builder| {
            use winit::platform::x11::EventLoopBuilderExtX11;
            builder.with_x11();
        }));
    }

    /// Whether `$DISPLAY` accepts a connection (it may not, e.g. without `XAUTHORITY`),
    /// tried the way winit's X11 backend connects.
    fn x11_reachable() -> bool {
        let Ok(xlib) = x11_dl::xlib::Xlib::open() else {
            return false;
        };
        // SAFETY: XOpenDisplay(NULL) connects to $DISPLAY and returns null on failure; a
        // successful connection is closed straight away.
        unsafe {
            let display = (xlib.XOpenDisplay)(std::ptr::null());
            if display.is_null() {
                return false;
            }
            (xlib.XCloseDisplay)(display);
        }
        true
    }
}
