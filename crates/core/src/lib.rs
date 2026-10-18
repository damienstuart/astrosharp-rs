//! Rust port of the AstroSharp image sharpening and denoising pipeline
//! (PSF, AstroClean and Hybrid models) from the original R/Shiny app.

pub mod color;
pub mod features;
pub mod fits;
pub mod image;
pub mod io;
pub mod model;
pub mod pipeline;
pub mod rround;
pub mod wcs;
pub mod weights;

pub use fits::FitsHeader;
pub use image::{Image, Plane};
pub use io::{BitDepth, FileFormat, SaveOptions, read_image, read_tiff, save_image, write_tiff};
pub use wcs::Wcs;

/// Version of AstroSharp (all crates share the workspace version).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub use pipeline::{
    Mode, ModelKind, NetKey, NetOutputs, Params, Progress, process, process_plane, process_with,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TIFF error: {0}")]
    Tiff(#[from] tiff::TiffError),
    #[error("PNG error: {0}")]
    Png(#[from] png::EncodingError),
    #[error("JPEG error: {0}")]
    Jpeg(#[from] jpeg_encoder::EncodingError),
    #[error("unsupported image: {0}")]
    Unsupported(String),
    #[error("{0}")]
    InvalidParams(String),
    #[error("invalid model weights: {0}")]
    BadWeights(String),
    #[error("cancelled")]
    Cancelled,
}
