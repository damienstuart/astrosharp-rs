#!/usr/bin/env Rscript
# Export the neuralnet weights used by AstroSharp (PSF + AstroClean models)
# into a compact binary format that the Rust core embeds.
#
# Usage: Rscript tools/export_weights.R <astrosharp_app_dir> <out_dir>
#   e.g. Rscript tools/export_weights.R ../AstroSharp/resources/app crates/core/weights
#
# File format (little endian):
#   magic   "ASNN"
#   u32     version (1)
#   u32     hidden activation (0 = relu, 1 = logistic)
#   u32     output activation (0 = relu, 1 = logistic)
#   u32     number of layers
#   per layer: u32 rows (inputs + 1, bias first), u32 cols (outputs)
#   per layer: rows*cols f64, row-major (row 0 is the bias row)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: export_weights.R <astrosharp_app_dir> <out_dir>")
app_dir <- normalizePath(args[1])
out_dir <- args[2]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

act_code <- function(f) {
  type <- attr(f, "type")
  switch(type, relu = 0L, logistic = 1L, stop("unsupported activation: ", type))
}

export_model <- function(rds, out) {
  m <- readRDS(rds)
  stopifnot(inherits(m, "nn"), !m$linear.output)
  w <- m$weights[[1]]
  stopifnot(!any(sapply(w, function(x) any(is.na(x)))))
  out_act <- if (is.null(m$output.act.fct)) m$act.fct else m$output.act.fct
  con <- file(out, "wb")
  on.exit(close(con))
  writeBin(charToRaw("ASNN"), con)
  writeBin(c(1L, act_code(m$act.fct), act_code(out_act), length(w)), con, size = 4, endian = "little")
  for (l in w) writeBin(c(nrow(l), ncol(l)), con, size = 4, endian = "little")
  for (l in w) writeBin(as.vector(t(l)), con, size = 8, endian = "little")
}

# The app builds the file name with paste0(), i.e. R's shortest number formatting.
for (v in seq(1, 8, by = 0.25)) {
  export_model(file.path(app_dir, "PSF", paste0("81_1_FWHM_", v, ".RDS")),
               file.path(out_dir, paste0("psf_", v, ".bin")))
}
export_model(file.path(app_dir, "AstroCleanv2NoBeta.RDS"), file.path(out_dir, "astroclean.bin"))

cat("exported", length(list.files(out_dir, pattern = "\\.bin$")), "models to", normalizePath(out_dir), "\n")
