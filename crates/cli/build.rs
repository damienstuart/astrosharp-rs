//! Embeds the app icon and version details in the Windows executable, where Explorer, the
//! taskbar and shortcuts show them. Nothing to do on other platforms.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rerun-if-changed=../../assets/AppIcon.ico");
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../assets/AppIcon.ico")
            .set("ProductName", "AstroSharp")
            .set("FileDescription", "AstroSharp command-line tool");
        res.compile()
            .expect("embedding the Windows icon and version details");
    }
}
