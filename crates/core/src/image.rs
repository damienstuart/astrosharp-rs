use std::ops::Range;

/// A grey (1 channel) or RGB (3 channel) image with values nominally in [0, 1],
/// stored row-major with interleaved channels.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub data: Vec<f64>,
}

impl Image {
    pub fn new(width: usize, height: usize, channels: usize, data: Vec<f64>) -> Self {
        assert!(
            channels == 1 || channels == 3,
            "images must have 1 or 3 channels"
        );
        assert_eq!(data.len(), width * height * channels);
        Image {
            width,
            height,
            channels,
            data,
        }
    }

    pub fn is_color(&self) -> bool {
        self.channels == 3
    }

    pub fn crop(&self, rows: Range<usize>, cols: Range<usize>) -> Image {
        assert!(rows.end <= self.height && cols.end <= self.width);
        let ch = self.channels;
        let mut data = Vec::with_capacity(rows.len() * cols.len() * ch);
        for r in rows.clone() {
            let start = (r * self.width + cols.start) * ch;
            data.extend_from_slice(&self.data[start..start + cols.len() * ch]);
        }
        Image::new(cols.len(), rows.len(), ch, data)
    }
}

/// A single-channel row-major buffer.
#[derive(Clone, Debug, PartialEq)]
pub struct Plane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl Plane {
    pub fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height);
        Plane {
            width,
            height,
            data,
        }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Plane::new(width, height, vec![0.0; width * height])
    }

    pub fn at(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.width + col]
    }

    pub fn set(&mut self, row: usize, col: usize, v: f64) {
        self.data[row * self.width + col] = v;
    }

    /// Copy surrounded by `pad` pixels of zeros on every side.
    pub fn padded(&self, pad: usize) -> Plane {
        let mut out = Plane::zeros(self.width + 2 * pad, self.height + 2 * pad);
        for r in 0..self.height {
            let dst = (r + pad) * out.width + pad;
            out.data[dst..dst + self.width].copy_from_slice(self.row(r));
        }
        out
    }

    pub fn sub(&self, rows: Range<usize>, cols: Range<usize>) -> Plane {
        let mut data = Vec::with_capacity(rows.len() * cols.len());
        for r in rows.clone() {
            data.extend_from_slice(&self.row(r)[cols.clone()]);
        }
        Plane::new(cols.len(), rows.len(), data)
    }

    /// Writes `values` (row-major, `rows.len() x cols.len()`) into the given region.
    pub fn write_region(&mut self, rows: Range<usize>, cols: Range<usize>, values: &[f64]) {
        assert_eq!(values.len(), rows.len() * cols.len());
        let w = cols.len();
        for (i, r) in rows.enumerate() {
            let dst = r * self.width + cols.start;
            self.data[dst..dst + w].copy_from_slice(&values[i * w..(i + 1) * w]);
        }
    }

    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.width..(r + 1) * self.width]
    }
}
