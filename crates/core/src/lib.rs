//! Rust port of the AstroSharp image sharpening and denoising pipeline
//! (PSF, AstroClean and Hybrid models) from the original R/Shiny app.

pub mod color;
pub mod features;
pub mod image;
pub mod io;
pub mod model;
pub mod pipeline;
pub mod rround;
pub mod weights;

pub use image::{Image, Plane};
pub use io::{BitDepth, FileFormat, SaveOptions, read_tiff, save_image, write_tiff};
pub use pipeline::{Mode, ModelKind, Params, Progress, process, process_plane};

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
