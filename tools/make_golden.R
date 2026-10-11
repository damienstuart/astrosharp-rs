#!/usr/bin/env Rscript
# Generate reference ("golden") outputs from the original AstroSharp R code so the
# Rust port can be checked for numerical parity.
#
# Usage: Rscript tools/make_golden.R <astrosharp_app_dir> <out_dir>
#   e.g. Rscript tools/make_golden.R ../AstroSharp/resources/app testdata/golden
#
# The app functions are sourced unmodified; Shiny's progress helpers are stubbed and
# the reactive arguments are passed as plain closures.
#
# Array file format (little endian):
#   magic "ASAR", u32 ndim, u32 dims[ndim] (R order: nrow, ncol[, 3]),
#   f64 data in row-major order ([row][col][channel]).

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: make_golden.R <astrosharp_app_dir> <out_dir>")
app_dir <- normalizePath(args[1])
dir.create(args[2], recursive = TRUE, showWarnings = FALSE)
out_dir <- normalizePath(args[2])

suppressMessages({
  library(tiff)
  library(neuralnet)
})

withProgress <- function(expr, ...) expr
incProgress <- function(...) invisible(NULL)

setwd(app_dir)
source("GetMatrixFun9x9.R")
source("First3BetasFun.R")
source("Combined_Sharpen_and_Clean.R")

write_arr <- function(x, name) {
  x <- as.array(x)
  d <- dim(x)
  v <- switch(length(d),
              x,
              as.vector(t(x)),
              as.vector(aperm(x, c(3, 2, 1))))
  con <- file(file.path(out_dir, name), "wb")
  on.exit(close(con))
  writeBin(charToRaw("ASAR"), con)
  writeBin(c(length(d), d), con, size = 4, endian = "little")
  writeBin(as.double(v), con, size = 8, endian = "little")
}

val <- function(v) {
  force(v)
  function() v
}

run_case <- function(image, color, model, psf, aggr, seg, preview, px, py) {
  input_file <- val(list(datapath = file.path(app_dir, image)))
  if (model == "Hybrid Model") {
    MixedPSFModel(val(model), val(psf), val(color), val(preview), input_file, val(aggr),
                  NULL, val(NULL), val(px), val(py), val(seg))
  } else {
    First3Betas(val(model), val(psf), val(color), val(preview), input_file, val(aggr),
                NULL, NULL, NULL, NULL, NULL, val(px), val(py), val(seg))
  }
}

cases <- read.table(header = TRUE, stringsAsFactors = FALSE, sep = "|", strip.white = TRUE, text = "
name                 | image   | color           | model                | psf  | aggr | seg | preview    | px   | py
gray_psf3            | T1.tif  | Black and White | PSF Model (Pre-Beta) | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
gray_psf1.25         | T1.tif  | Black and White | PSF Model (Pre-Beta) | 1.25 | 0.3  | 325 | Full Image | 0.5  | 0.5
gray_psf8            | T1.tif  | Black and White | PSF Model (Pre-Beta) | 8    | 1    | 150 | Full Image | 0.5  | 0.5
gray_astroclean      | T1.tif  | Black and White | AstroClean           | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
gray_hybrid_s325     | T1.tif  | Black and White | Hybrid Model         | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
gray_hybrid_s150     | T1.tif  | Black and White | Hybrid Model         | 3    | 0.66 | 150 | Full Image | 0.5  | 0.5
gray_hybrid_s169     | T1.tif  | Black and White | Hybrid Model         | 2.5  | 0.4  | 169 | Full Image | 0.5  | 0.5
gray_hybrid_s176     | T1.tif  | Black and White | Hybrid Model         | 5.75 | 0.9  | 176 | Full Image | 0.5  | 0.5
gray_psf3_preview    | T1.tif  | Black and White | PSF Model (Pre-Beta) | 3    | 0.66 | 325 | Preview    | 0.37 | 0.61
gray_hybrid_preview  | T1.tif  | Black and White | Hybrid Model         | 4    | 0.5  | 50  | Preview    | 0.5  | 0.5
gray16_psf2          | R1.tif  | Black and White | PSF Model (Pre-Beta) | 2    | 0.8  | 325 | Full Image | 0.5  | 0.5
color_psf3           | CT1.tif | Color           | PSF Model (Pre-Beta) | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
color_astroclean     | CT1.tif | Color           | AstroClean           | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
color_hybrid_s325    | CT1.tif | Color           | Hybrid Model         | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5
color_psf3_preview   | CT1.tif | Color           | PSF Model (Pre-Beta) | 3    | 0.66 | 325 | Preview    | 0.45 | 0.55
")
write.table(cases, file.path(out_dir, "cases.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# Inputs exactly as readTIFF() sees them (checks the Rust TIFF reader).
for (img in unique(c(cases$image, "CR1.tif"))) write_arr(readTIFF(img), paste0("input_", img, ".bin"))

for (k in seq_len(nrow(cases))) {
  cs <- cases[k, ]
  t <- system.time(res <- run_case(cs$image, cs$color, cs$model, cs$psf, cs$aggr, cs$seg, cs$preview, cs$px, cs$py))
  write_arr(res, paste0(cs$name, ".bin"))
  cat(sprintf("%-22s %-12s %6.1fs\n", cs$name, paste(dim(res), collapse = "x"), t[3]))
}

# Feature order of getmatrix9: encode (row, col) into the pixel value and decode.
probe <- outer(1:9, 1:9, function(r, c) r * 100 + c)
feat <- getmatrix9(probe)
writeLines(sprintf("%d %d", feat %/% 100 - 5, feat %% 100 - 5), file.path(out_dir, "offsets_9x9.txt"))

# Colour conversion vectors.
set.seed(42)
edge <- rbind(c(0, 0, 0), c(1, 1, 1), c(1, 0, 0), c(0, 1, 0), c(0, 0, 1), c(1e-6, 1e-6, 1e-6),
              c(0.04045, 0.04045, 0.04045), c(0.04044, 0.5, 0.04046), c(0.5, 0.5, 0.5))
rgb <- rbind(edge, matrix(runif(3 * 20000), ncol = 3))
luv <- convertColor(rgb, from = "sRGB", to = "Luv")
luv_mod <- luv
luv_mod[, 1] <- c(0, 100, 8, 7.99, 8.01, runif(nrow(luv) - 5, 0, 100))
rgb_back <- convertColor(luv_mod, from = "Luv", to = "sRGB")
write_arr(rgb, "color_rgb.bin")
write_arr(luv, "color_luv.bin")
write_arr(luv_mod, "color_luv_mod.bin")
write_arr(rgb_back, "color_rgb_back.bin")

# round(x, 5) as used by convertColor's trim().
x <- c(runif(50000, -0.2, 1.2), (0:100000) / 1e5 + 5e-6, -((0:2000) / 1e5 + 5e-6))
write_arr(cbind(x, round(x, 5)), "round5.bin")

cat("golden data written to", out_dir, "\n")
