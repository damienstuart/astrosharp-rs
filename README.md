# astrosharp-rs

A Rust port of [AstroSharp](https://github.com/deepskydetail/AstroSharp)'s image processing.
The original is an R/Shiny app packaged with Electron. This port covers the **PSF**
(sharpen), **AstroClean** (denoise) and **Hybrid** (PSF then AstroClean) models.
The legacy "First Beta" and "Second Beta" models are not ported.
It reads TIFF and FITS (Siril) images and saves TIFF, FITS, PNG or JPEG.

The trained `neuralnet` weights are exported from the original `.RDS` files and embedded
in the binaries, so nothing else is needed at runtime: no R, no Electron.

> The upstream repository has no license covering its code or model weights. This port
> is for personal use; get the author's permission before redistributing it.

## Recommended workflow

AstroSharp is meant as a **final touch-up on a stretched (non-linear) image**. The models were trained on stretched data, so stretch first (in Siril, for example) and run AstroSharp last.

Images that look linear trigger a note in the GUI and on the CLI's stderr: the warning fires when 99% of pixels are darker than 0.15. Linear stacks typically sit at or below 0.04, and stretched images at 0.5 or above. Processing still runs, and linear images also look nearly black in the GUI.

## Layout

| Path | Contents |
|---|---|
| `crates/core` | Library: MLP inference, 9x9 feature windows, pipeline, sRGB↔Luv, TIFF I/O, embedded weights |
| `crates/cli` | `astrosharp` command-line tool |
| `crates/gui` | `astrosharp-gui` desktop app (egui/eframe) |
| `crates/telemetry` | Tracing setup shared by the CLI and GUI: stderr logging and OpenTelemetry export |
| `tools/export_weights.R` | Converts the `.RDS` models to `crates/core/weights/*.bin` |
| `tools/make_golden.R` | Runs the original R functions headless to produce reference outputs |
| `tools/bundle_macos.sh` | Builds an unsigned `AstroSharp.app` |
| `tools/icon` | Standalone helper (uses `resvg`) that renders the icon SVG for `make icon` |
| `assets/` | App icon: `icon.svg` (source), plus the generated `AppIcon.icns` and `icon-512.png` (window/Dock icon) |
| `testdata/samples` | The sample TIFFs shipped with the R app |

## Usage

`make` lists the common tasks:

```sh
make run [IMAGE=image.tif]   # desktop app
make release                 # build target/release/astrosharp and astrosharp-gui
make app                     # target/release/bundle/AstroSharp.app (unsigned)
make install                 # ...and copy it to /Applications (APP_DIR=... to change)
make check                   # rustfmt check + clippy + tests
make icon                    # re-render assets/icon.svg into AppIcon.icns + icon-512.png (commit the results)
make golden                  # regenerate R reference outputs (R_APP=... if not ../AstroSharp/resources/app)
```

The app bundle also contains the CLI, at `/Applications/AstroSharp.app/Contents/MacOS/astrosharp-cli`
after `make install` (the Siril script uses it from there).

CLI example: `astrosharp in.fit out.fit --model hybrid --psf 3 --aggressiveness 0.66`.
The input can be TIFF or FITS (`.fit`/`.fits`/`.fts`).
Options:
- `--model hybrid|psf|astroclean` (default `hybrid`)
- `--psf <1..8 step 0.25>`
- `--aggressiveness <0..1>`
- `--mode seamless|parity`
- `--chunk-size <n>` (Hybrid in parity mode only)
- Output format follows the file extension: `.tif`/`.tiff`, `.fit`/`.fits`/`.fts`, `.png` or `.jpg`/`.jpeg`
- `--depth <8|16|32>`:
  - TIFF: 8, 16 (default) or 32 (float).
  - FITS: 16 or 32. The default is the input FITS's depth, otherwise 32.
  - PNG: 8 or 16 (default).
  - JPEG: 8 only.
- `--quality <1..100>` (JPEG only, default 90)

GUI (Hybrid is the default model):
- **Processing** starts with "Process image"; opening a file never processes it. A progress bar with Cancel shows while it runs.
  - The buttons are colour-coded by what they do: blue processes, amber discards the result, red discards the result and the settings.
- **Live update** (off by default) reprocesses about 0.25 s after the last settings change. With it off, nothing is processed until you press a button.
- **Navigation:** drag to pan, scroll or pinch to zoom, double-click to fit. Hold Space to show the original.
- **Compare before/after** (off by default) splits the view into original | processed. Drag the divider's handle (or the line) to move it.
- **Preview region only** (off by default) processes just a region while you adjust settings. Choose it with Shift-drag or right-drag, and process the whole image with "Process full image". "Update preview" refreshes the region; with Live update on it also refreshes automatically.
- **Open image…** (or drop a file on the window) accepts TIFF and FITS.
- **Reset image** and **Reset all** sit under the processing buttons. Both discard the processed result, cancelling any processing in progress, and reload the image from disk, for example after you've changed it in another program. If the file can no longer be read, the image as first loaded is shown.
  - **Reset image** keeps the model settings. The zoom and preview region are kept too, unless the image's size changed.
  - **Reset all** also restores the default model settings (Hybrid, PSF 3, Aggressiveness 0.66, Seamless) and a fresh view.
  - Both keep the view options and the Save as choice. With Live update on, neither reprocesses until you change a setting.
- **Save as** picks the output format:
  - TIFF (default): 16-bit, or 32-bit float for further processing.
  - FITS: 32-bit float or 16-bit. This is selected automatically, at the source's depth, when you open a FITS file.
  - PNG: lossless, 16-bit (default) or 8-bit.
  - JPEG: quality 50–100 (default 90), written without chroma subsampling to keep small coloured stars crisp.

  Saving runs in the background.
- **Quit** closes the app.

## FITS (Siril)

- **Supported files:** uncompressed FITS with the image in the primary HDU, grey or RGB (three planes). Samples can be 8/16/32/64-bit integer or 32/64-bit float, which covers what Siril writes.
- **Not supported:** compressed FITS (`.fz`) and images stored only in extensions.
- **Values:** integer data is scaled by its type's range (after `BZERO`/`BSCALE`). Float data is used as stored, since Siril keeps it in 0–1; float data clearly outside that range is scaled down and restored on save. Blank (NaN) pixels become 0.
- **Row order:** follows `ROWORDER`, defaulting to FITS's bottom-up. Images are processed top row first, like TIFFs, and written back in the source's row order.
- **Header:** saving as FITS from a FITS source keeps every header card verbatim and in order, except for the following:
  - Structural keywords (`SIMPLE`, `BITPIX`, `NAXISn`, `EXTEND`, `BZERO`, `BSCALE`) are regenerated.
  - Values made stale by processing (`CHECKSUM`, `DATASUM`, `DATAMIN`, `DATAMAX`, `BLANK`) are dropped.
  - `DATE` is refreshed.
  - A `HISTORY AstroSharp <version>: <settings>` line is appended.

  Plate solutions (WCS/SIP), observation details and history are therefore kept, and any HDUs after the image are copied unchanged.
- **Linear data:** see [Recommended workflow](#recommended-workflow).

## Tracing (OpenTelemetry)

The CLI and GUI can send traces of what they do to an OpenTelemetry endpoint such as
[VictoriaTraces](https://docs.victoriametrics.com/victoriatraces/). Nothing is sent unless an endpoint is configured.

- **Configuration:** set the standard variables in the environment, or in `~/.config/astrosharp/otel.env` (under `$XDG_CONFIG_HOME` if that's set).
  - Apps launched from Finder don't see shell variables, so use the file for AstroSharp.app, and for the bundled CLI when Siril runs it.
  - An endpoint in the environment overrides the file's.

  ```sh
  # ~/.config/astrosharp/otel.env (KEY=value lines; # starts a comment)
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:10428/insert/opentelemetry
  ```
- **Settings:**
  - `OTEL_EXPORTER_OTLP_ENDPOINT` is a base URL; `/v1/traces` is appended. `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` is a full URL, used as is.
  - `OTEL_EXPORTER_OTLP_HEADERS` (`key=value,key=value`) and `OTEL_EXPORTER_OTLP_TIMEOUT` (milliseconds, default 5000).
  - `OTEL_SDK_DISABLED=true` turns export off.
  - `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES` work from the environment only.
- **Transport:** OTLP over plain `http://`, protobuf-encoded. HTTPS and gRPC aren't built in.
- **Services:** `astrosharp-cli` and `astrosharp-gui`, tagged with `service.version`.
- **Spans** cover each operation and processing stage:
  - The CLI's `run` span has `read_image`, `process` and `save_image` as children.
  - In the GUI, `open_image`, `process_job` (full image or preview, with an `outcome` of ok, cancelled or failed), `save_job` and `reset` (image or all) each start their own trace.
  - Under `process`: `to_luv` and `from_luv` (colour only), `psf_pass` and `astroclean_pass`, or `hybrid_parity` in R parity mode.
  - Failures mark the top span as an error, with the message.
- **Detail:** `ASTROSHARP_TRACE` filters what is exported, using [`EnvFilter`](https://docs.rs/tracing-subscriber/latest/tracing_subscriber/filter/struct.EnvFilter.html) syntax (default `astrosharp=info`).
  - `ASTROSHARP_TRACE=astrosharp=debug` adds a span per parity-mode chunk and FITS header details.
  - It can also go in `otel.env`.
- **Logging:** warnings and errors, including failed exports, go to stderr; `RUST_LOG` changes the level (default `warn`).
- **Privacy:** traces include the full paths of the files you open and save.

## Fidelity to the R app

`cargo test --release` checks the port against reference outputs generated by the R code:

```sh
Rscript tools/make_golden.R ../AstroSharp/resources/app testdata/golden   # once, ~25 s
cargo test --release
```

If `testdata/golden` is missing, the R-comparison tests print `SKIPPED` and pass. The tests cover:

- **Processed values:** match R within 2.4e-15 (f64 rounding noise) for every model, both grey and colour, several chunk sizes and preview crops.
- **Colour output:** matches R exactly.
- **16-bit TIFFs:** identical to the R app's `writeTIFF` output, sample for sample.
- **Building blocks:** `readTIFF` scaling, R's `round(x, 5)` and the sRGB↔Luv conversion are bit-identical.

Behaviours deliberately reproduced:
- **Unprocessed border.** Only pixels 5..n-5 (1-based) are processed; a 4 px border at the start and 5 px at the end keep their input values.
- **Feature order.** The 9x9 features use `getmatrix9`'s irregular column order, which the models were trained on.
- **Colour.** Colour images go through R's sRGB↔Luv (R's own D65 white point), only L is processed, and the output is rounded to 5 decimals.

Modes:
- **`parity`** reproduces the R app's chunked Hybrid processing, including faint seams at chunk boundaries: stage 2 sees unsharpened pixels at each chunk's rim.
- **`seamless`** (default) processes the whole image in two passes instead, and keeps colour output at full precision (R rounds it to 5 decimals, which would cost precision in 32-bit linear data). For grey PSF and AstroClean, both modes give identical output.

Intentional differences from the R app:
- **Colour vs grey** is detected from the file. The R app needed the correct radio button and failed otherwise.
- **Preview** uses any region you choose and processes it with an 8 px margin, so it matches full-image output. The R app processed a fixed 10% crop as a standalone image.
- **Out-of-range values** saturate when written to 16-bit; R's `writeTIFF` wraps them.
- **Alpha channels** are dropped instead of failing.
- **TIFF tags.** Output files carry a `Predictor=1` tag. libtiff in R logs a harmless "unknown field 317" warning when reading them.

## Performance

Apple M5 Max, 6000x4000 (24 MP) image:

| | R app (single core) | Rust |
|---|---|---|
| PSF, grey | 110 s | 0.95 s |
| Hybrid, grey | not run at 24 MP (34.5 s at 4 MP) | 2.0 s (2.9 s parity; 0.32 s at 4 MP) |
| PSF / Hybrid, RGB | — | 1.2 s / 2.2 s |
