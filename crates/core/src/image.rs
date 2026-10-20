use std::ops::Range;

use rayon::prelude::*;

/// 99th-percentile brightness below which an image is treated as linear (unstretched).
/// Linear stacks tested sit at or below 0.04, stretched images at 0.58 or above.
const LINEAR_P99: f64 = 0.15;

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

    /// Whether the image looks linear (unstretched): 99% of pixels darker than 0.15.
    /// AstroSharp's models were trained on stretched images, so it is meant for those.
    pub fn looks_linear(&self) -> bool {
        let pixels = self.width * self.height;
        let step = (pixels / 1_000_000).max(1);
        let mut lum: Vec<f64> = self
            .data
            .chunks_exact(self.channels)
            .step_by(step)
            .map(|p| p.iter().sum::<f64>() / p.len() as f64)
            .collect();
        if lum.is_empty() {
            return false;
        }
        let k = (lum.len() - 1) * 99 / 100;
        let (_, p99, _) = lum.select_nth_unstable_by(k, f64::total_cmp);
        *p99 < LINEAR_P99
    }

    /// Histogram of the samples (all channels pooled) in `bins` equal bins over [0, 1],
    /// normalised so the fullest bin is 1, as the R app plots it. Values outside [0, 1]
    /// count in the end bins.
    pub fn histogram(&self, bins: usize) -> Vec<f32> {
        assert!(bins > 0);
        let counts = self
            .data
            .par_chunks(1 << 16)
            .fold(
                || vec![0u64; bins],
                |mut acc, chunk| {
                    for &v in chunk {
                        // R's hist() closes bins on the right: (a, b], with 0 in the first.
                        let b = (v * bins as f64).ceil() as isize - 1;
                        acc[b.clamp(0, bins as isize - 1) as usize] += 1;
                    }
                    acc
                },
            )
            .reduce(
                || vec![0u64; bins],
                |mut a, b| {
                    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                    a
                },
            );
        let max = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
        counts.into_iter().map(|c| c as f32 / max).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Dark background with 0.5% bright "stars", like a linear stack.
    fn linear_like(w: usize, h: usize) -> Image {
        let data = (0..w * h)
            .map(|i| {
                if i % 200 == 0 {
                    0.9
                } else {
                    0.01 + (i % 7) as f64 * 0.001
                }
            })
            .collect();
        Image::new(w, h, 1, data)
    }

    #[test]
    fn histogram_bins_are_closed_on_the_right() {
        let img = Image::new(5, 1, 1, vec![0.0, 0.1, 0.10001, 1.0, 2.0]);
        let h = img.histogram(10);
        // 0 and 0.1 fall in the first bin, 0.10001 in the second, 1 and 2 in the last.
        assert_eq!(h, [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn linear_detection() {
        assert!(linear_like(100, 80).looks_linear());
        let stretched = Image::new(10, 10, 1, (0..100).map(|i| i as f64 / 99.0).collect());
        assert!(!stretched.looks_linear());
        // Dark background but a stretched top end (like a final image of a dark field).
        let dark_field = Image::new(
            10,
            10,
            1,
            (0..100).map(|i| if i < 90 { 0.05 } else { 0.6 }).collect(),
        );
        assert!(!dark_field.looks_linear());
    }
}
