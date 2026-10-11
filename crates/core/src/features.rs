//! The 81-pixel neighbourhood fed to the 9x9 models (`getmatrix9` in `GetMatrixFun9x9.R`).

use crate::image::Plane;

/// Pixels on each side of the centre that a window needs.
pub const RADIUS: usize = 4;
pub const N_FEATURES: usize = 81;
/// Index of the centre pixel (`v9` in the R code).
pub const CENTER: usize = 8;

/// `(row, col)` offsets in the exact column order of `getmatrix9`. The order is irregular
/// (index 55 is `(+4, +4)`, ahead of the rest of the `+1..+4` columns), and the trained
/// models depend on it.
#[rustfmt::skip]
pub const OFFSETS_9X9: [(i8, i8); N_FEATURES] = [
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0),
    (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, -2), (-1, 2), (0, -2), (0, 2),
    (1, -2), (1, 2), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2),
    (-4, -4), (-3, -4), (-2, -4), (-1, -4), (0, -4), (1, -4), (2, -4), (3, -4), (4, -4),
    (-4, -3), (-3, -3), (-2, -3), (-1, -3), (0, -3), (1, -3), (2, -3), (3, -3), (4, -3),
    (-4, -2), (-3, -2), (3, -2), (4, -2),
    (-4, -1), (-3, -1), (3, -1), (4, -1),
    (-4, 0), (-3, 0), (3, 0), (4, 0),
    (4, 4),
    (-4, 1), (-3, 1), (3, 1), (4, 1),
    (-4, 2), (-3, 2), (3, 2), (4, 2),
    (-4, 3), (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
    (-4, 4), (-3, 4), (-2, 4), (-1, 4), (0, 4), (1, 4), (2, 4), (3, 4),
];

/// Writes the features of pixels `(row, cols)` into `out` (`cols.len() x 81`, row-major).
/// Every window must lie inside `plane`.
pub fn fill_row(plane: &Plane, row: usize, cols: std::ops::Range<usize>, out: &mut [f64]) {
    debug_assert!(row >= RADIUS && row + RADIUS < plane.height);
    debug_assert!(cols.start >= RADIUS && cols.end + RADIUS <= plane.width);
    debug_assert_eq!(out.len(), cols.len() * N_FEATURES);
    let w = plane.width as isize;
    let offsets: [isize; N_FEATURES] = OFFSETS_9X9.map(|(dr, dc)| dr as isize * w + dc as isize);
    let base = (row * plane.width) as isize;
    for (c, feats) in cols.zip(out.as_chunks_mut::<N_FEATURES>().0) {
        let centre = base + c as isize;
        for (f, off) in feats.iter_mut().zip(offsets) {
            *f = plane.data[(centre + off) as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_a_permutation_of_the_9x9_window() {
        let mut seen = [[false; 9]; 9];
        for (dr, dc) in OFFSETS_9X9 {
            let cell = &mut seen[(dr + 4) as usize][(dc + 4) as usize];
            assert!(!*cell, "duplicate offset ({dr}, {dc})");
            *cell = true;
        }
        assert_eq!(OFFSETS_9X9[CENTER], (0, 0));
    }
}
