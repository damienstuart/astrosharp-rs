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

# ---- Star mask, Dual PSF and Star Mask models (DualPSF branch: DualPSF.R, StarMask.R) ----
# Loaded last: dplyr masks a few base functions (the code above only uses neuralnet::compute).
suppressMessages({
  library(imager)
  library(magrittr)
  library(dplyr)
})
source("DualPSF.R")
source("StarMask.R")

writeLines(c(R.version.string, paste("imager", packageVersion("imager")),
             paste("machine", Sys.info()[["machine"]])), file.path(out_dir, "versions.txt"))

run_mask_case <- function(cs) {
  input_file <- val(list(datapath = file.path(app_dir, cs$image)))
  if (cs$model == "Dual PSF") {
    DualPSF(val(cs$model), val(cs$psf), val(cs$psf2), val(cs$color), val(cs$preview), input_file,
            val(cs$aggr), NULL, val(NULL), val(cs$px), val(cs$py), val(cs$feather), val(cs$blend),
            val(cs$blend2), val(cs$seg), val(cs$sensi), val(cs$fix))
  } else {
    starmask(val(cs$feather), val(cs$blend), val(cs$blend2), val(cs$fix), input_file, val(cs$px),
             val(cs$py), val(cs$seg), val(cs$preview), val(cs$color), val(cs$sensi))
  }
}

mask_cases <- read.table(header = TRUE, stringsAsFactors = FALSE, sep = "|", strip.white = TRUE, text = "
name             | image   | color           | model     | psf | psf2 | aggr | seg | preview    | px   | py   | sensi | fix | feather | blend | blend2
dual_s325        | T1.tif  | Black and White | Dual PSF  | 3   | 2    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
dual_s100        | T1.tif  | Black and White | Dual PSF  | 4   | 2.5  | 0.8  | 100 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
dual_s50_params  | T1.tif  | Black and White | Dual PSF  | 3   | 1.5  | 1    | 50  | Full Image | 0.5  | 0.5  | 95    | 0   | 2.5     | 2     | 20
dual_same_psf    | T1.tif  | Black and White | Dual PSF  | 3   | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
dual_deriche     | T1.tif  | Black and White | Dual PSF  | 3   | 2    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 0.3     | 9     | 100
dual_preview     | T1.tif  | Black and White | Dual PSF  | 3   | 2    | 0.66 | 25  | Preview    | 0.37 | 0.61 | 98    | 2   | 5       | 9     | 100
dual_r1          | R1.tif  | Black and White | Dual PSF  | 2.5 | 1.75 | 0.9  | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
dual_color       | CT1.tif | Color           | Dual PSF  | 3   | 2    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
dual_color_prev  | CT1.tif | Color           | Dual PSF  | 3   | 2    | 0.66 | 30  | Preview    | 0.45 | 0.55 | 98    | 2   | 5       | 9     | 100
mask_default     | T1.tif  | Black and White | Star Mask | 3   | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
mask_params      | T1.tif  | Black and White | Star Mask | 3   | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 50    | 0   | 1       | 1     | 1
mask_noblur      | T1.tif  | Black and White | Star Mask | 3   | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 0.05    | 9     | 100
mask_preview     | T1.tif  | Black and White | Star Mask | 3   | 3    | 0.66 | 325 | Preview    | 0.37 | 0.61 | 98    | 2   | 5       | 9     | 100
mask_color       | CT1.tif | Color           | Star Mask | 3   | 3    | 0.66 | 325 | Full Image | 0.5  | 0.5  | 98    | 2   | 5       | 9     | 100
")
write.table(mask_cases, file.path(out_dir, "mask_cases.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

for (k in seq_len(nrow(mask_cases))) {
  cs <- mask_cases[k, ]
  t <- system.time(res <- run_mask_case(cs))
  write_arr(res, paste0(cs$name, ".bin"))
  cat(sprintf("%-22s %-12s %6.1fs\n", cs$name, paste(dim(res), collapse = "x"), t[3]))
}

# Star mask stages. Besides imager's results, plain-R transcriptions of CImg's blur and
# Hessian ("nofma"): R's arithmetic is never fused, so the Rust port must match these bit
# for bit, while imager itself may be compiled with fused multiply-adds.
f32 <- function(x) readBin(writeBin(x, raw(), size = 4), "double", size = 4)
vv_coefs <- function(s) {
  m0 <- 1.16680; m1 <- 1.10783; m2 <- 1.40586; m1sq <- m1*m1; m2sq <- m2*m2
  q <- if (s < 3.556) -0.2568 + 0.5784*s + 0.0561*s*s else 2.5091 + 0.9804*(s - 3.556)
  qsq <- q*q; scale <- (m0 + q)*(m1sq + m2sq + 2*m1*q + qsq)
  b1 <- -q*(2*m0*m1 + m1sq + m2sq + (2*m0 + 4*m1)*q + 3*qsq)/scale
  b2 <- qsq*(m0 + 2*m1 + 3*q)/scale; b3 <- -qsq*q/scale; B <- (m0*(m1sq + m2sq))/scale
  f <- c(B, -b1, -b2, -b3); a1 <- f[2]; a2 <- f[3]; a3 <- f[4]
  sM <- 1/((1 + a1 - a2 + a3)*(1 - a1 - a2 - a3)*(1 + a2 + (a1 - a3)*a3))
  M <- c(sM*(-a3*a1 + 1 - a3*a3 - a2), sM*(a3 + a1)*(a2 + a3*a1), sM*a3*(a1 + a3*a2),
         sM*(a1 + a3*a2), -sM*(a2 - 1)*(a2 + a3*a1), -sM*a3*(a3*a1 + a3*a3 + a2 - 1),
         sM*(a3*a1 + a2 + a1*a1 - a2*a2), sM*(a1*a2 + a3*a2*a2 - a1*a3*a3 - a3*a3*a3 - a3*a2 + a3),
         sM*a3*(a1 + a3*a2))
  list(f = f, M = M)
}
vv_line <- function(x, k) {
  f <- k$f; M <- k$M; N <- length(x); sumsq <- f[1]; sum <- sumsq*sumsq; a1 <- f[2]; a2 <- f[3]; a3 <- f[4]
  iplus <- x[N]; v1 <- v2 <- v3 <- x[1]/sumsq
  for (n in 1:N) { v <- x[n]; v <- v + v1*a1; v <- v + v2*a2; v <- v + v3*a3; x[n] <- v; v3 <- v2; v2 <- v1; v1 <- v }
  up <- iplus/(1 - a1 - a2 - a3); vp <- up/(1 - a1 - a2 - a3)
  u0 <- v1 - up; u1 <- v2 - up; u2 <- v3 - up
  w0 <- (M[1]*u0 + M[2]*u1 + M[3]*u2 + vp)*sum; w1 <- (M[4]*u0 + M[5]*u1 + M[6]*u2 + vp)*sum
  w2 <- (M[7]*u0 + M[8]*u1 + M[9]*u2 + vp)*sum
  x[N] <- w0; v1 <- w0; v2 <- w1; v3 <- w2
  for (n in (N-1):1) { v <- x[n]*sum; v <- v + v1*a1; v <- v + v2*a2; v <- v + v3*a3; x[n] <- v; v3 <- v2; v2 <- v1; v1 <- v }
  x
}
dr_coefs <- function(s) {
  alpha <- f32(1.695)/s; ema <- exp(-alpha); ema2 <- exp(-2*alpha); b1 <- -2*ema; b2 <- ema2
  k <- (1 - ema)*(1 - ema)/(1 + 2*alpha*ema - ema2)
  a0 <- k; a1 <- k*(alpha - 1)*ema; a2 <- k*(alpha + 1)*ema; a3 <- -k*ema2
  list(a0 = a0, a1 = a1, a2 = a2, a3 = a3, b1 = b1, b2 = b2,
       coefp = (a0 + a1)/(1 + b1 + b2), coefn = (a2 + a3)/(1 + b1 + b2))
}
dr_line <- function(x, k) {
  N <- length(x); y <- numeric(N); xp <- x[1]; yb <- yp <- k$coefp*xp
  for (m in 1:N) { xc <- x[m]; yc <- k$a0*xc + k$a1*xp - k$b1*yp - k$b2*yb; y[m] <- yc; xp <- xc; yb <- yp; yp <- yc }
  xn <- xa <- x[N]; yn <- ya <- k$coefn*xn
  for (n in N:1) { xc <- x[n]; yc <- k$a2*xn + k$a3*xa - k$b1*yn - k$b2*ya; xa <- xn; xn <- xc; ya <- yn; yn <- yc; x[n] <- y[n] + yc }
  x
}
nofma_blur <- function(m, sigma) {
  s <- f32(sigma)
  if (s < f32(0.1)) return(m)
  if (s < 0.5) { k <- dr_coefs(s); line <- dr_line } else { k <- vv_coefs(s); line <- vv_line }
  if (nrow(m) > 1) m <- apply(m, 2, line, k = k)
  if (ncol(m) > 1) m <- t(apply(m, 1, line, k = k))
  m
}
nofma_hess <- function(g) {
  h <- nrow(g); w <- ncol(g)
  xx <- g; xx[2:(h-1),] <- (g[3:h,] + g[1:(h-2),]) - 2*g[2:(h-1),]; xx[1,] <- g[2,] - g[1,]; xx[h,] <- g[h-1,] - g[h,]
  yy <- g; yy[,2:(w-1)] <- (g[,3:w] + g[,1:(w-2)]) - 2*g[,2:(w-1)]; yy[,1] <- g[,2] - g[,1]; yy[,w] <- g[,w-1] - g[,w]
  rn <- c(2:h, h); rp <- c(1, 1:(h-1)); cn <- c(2:w, w); cp <- c(1, 1:(w-1))
  xy <- (((g[rn, cn] + g[rp, cp]) - g[rn, cp]) - g[rp, cn])/4
  list(xx = xx, yy = yy, xy = xy)
}
nofma_D <- function(m) {
  D <- NULL
  for (s in seq(2, 20, l = 10)) {
    h <- nofma_hess(nofma_blur(m, s))
    d <- s^2*(h$xx*h$yy - h$xy^2)
    D <- if (is.null(D)) d else pmax(D, d)
  }
  D
}
hessdet <- function(im, scale) isoblur(im, scale) %>% imhessian %$% { scale^2*(xx*yy - xy^2) }

crop <- readTIFF("T1.tif")[101:260, 121:270]
write_arr(crop, "sm_crop.bin")
blur_sigmas <- c(0.05, 0.3, 0.5, 2, 3.556, 4, 5, 20)
write_arr(blur_sigmas, "sm_blur_sigmas.bin")
for (s in blur_sigmas) {
  write_arr(as.matrix(isoblur(as.cimg(crop), s)), sprintf("sm_blur_%s.bin", s))
  write_arr(nofma_blur(crop, s), sprintf("sm_blur_nofma_%s.bin", s))
}
g4 <- isoblur(as.cimg(crop), 4)
hs <- imhessian(g4)
write_arr(as.matrix(hs$xx), "sm_hess_xx.bin")
write_arr(as.matrix(hs$xy), "sm_hess_xy.bin")
write_arr(as.matrix(hs$yy), "sm_hess_yy.bin")
write_arr(nofma_D(crop), "sm_crop_D_nofma.bin")

t1 <- readTIFF("T1.tif")
D <- map_il(seq(2, 20, l = 10), function(s) hessdet(as.cimg(t1), s)) %>% parmax
write_arr(as.matrix(D), "sm_t1_D.bin")
sensis <- c(98, 90, 50, 98.5)
write_arr(cbind(sensis, sapply(sensis, function(q) attr(threshold(D, paste0(q, "%")), "threshold"))),
          "sm_t1_thresholds.bin")
lab <- as.matrix(label(threshold(D, "98%")))
write_arr(lab, "sm_t1_label98.bin")
Bb <- as.matrix(isoblur(as.cimg(ifelse(lab > 2, 1, 0)), 5))
write_arr(Bb, "sm_t1_Bb.bin")
write_arr(pbeta(Bb, 9, 100), "sm_t1_M.bin")

xs <- c(1e-300, 1e-20, 1e-10, 1e-5, seq(0.001, 0.999, length.out = 600), 1 - 1e-10)
pairs <- rbind(c(9, 100), c(1e-4, 1e-4), c(0.5, 0.5), c(1, 1), c(2, 3), c(100, 9), c(50, 500),
               c(1000, 1000), c(0.05, 100), c(2, 20), c(9, 1), c(0.0001, 100))
write_arr(do.call(rbind, lapply(seq_len(nrow(pairs)), function(k)
  cbind(xs, pairs[k, 1], pairs[k, 2], pbeta(xs, pairs[k, 1], pairs[k, 2])))), "pbeta_grid.bin")

ns <- c(1001, 1369, 260500, 390600, 24000000)
write_arr(cbind(ns, t(sapply(ns, function(n) round(seq(1, n, l = 1000))))), "sample_idx.bin")

cat("golden data written to", out_dir, "\n")
