# AstroSharp for Siril

A Siril script that runs [AstroSharp](../astrosharp-rs) on the image currently open in
Siril. It can sharpen (PSF), denoise (AstroClean) or do both (Hybrid).

You can preview the result and compare it with the original, using zoom, pan and a
draggable before/after divider like the AstroSharp app. When you're happy, **Send to
Siril** updates the image in place with an undo point, so **Edit → Undo** (Ctrl/Cmd+Z)
restores it. The settings are recorded in the image's FITS HISTORY, for example
`AstroSharp: Hybrid PSF 3 aggr 0.66`.

AstroSharp is meant as a **final touch-up on a stretched (non-linear) image**. If the image
looks linear, the script logs a warning in Siril's console, but it still processes the image.

## Requirements

- Siril 1.4 or later. The script uses Siril's bundled Python with numpy and PyQt6.
- AstroSharp 0.3.0 or later. Run `make install` in `astrosharp-rs`, which puts
  `AstroSharp.app`, with its CLI at `Contents/MacOS/astrosharp-cli`, in `/Applications`.
  The script finds it there; otherwise choose the CLI, or `AstroSharp.app` itself, with
  **Browse…**.

## Install

Copy `AstroSharp.py` into a folder on Siril's script path, for example `~/.siril/scripts`:

```sh
make install                       # copies to ~/.siril/scripts (SIRIL_SCRIPTS=... to change)
```

Then refresh Siril's script list (**Scripts → Refresh Script List**, or restart Siril).
The script appears in the **Scripts** menu.

## Use

- **Scripts → AstroSharp** opens a window with the settings on the left and the image on
  the right:
  - **Settings:** choose the model, PSF FWHM, Aggressiveness and, for Hybrid, the mode.
  - **Preview** processes a copy of the image; Siril's image is not changed. **Live
    preview** (off by default) refreshes it whenever a setting changes.
  - **Viewing:** scroll to zoom, drag to pan, double-click to fit, and hold Space to see
    the original. **Compare before/after** adds a divider; drag its handle to move it.
  - **Reset** discards the preview and reloads the image from Siril, for example after
    you've changed it there.
  - **Send to Siril** applies the previewed result with an undo point. If the settings
    changed since the preview, it processes first. If the image in Siril changed after
    the preview (for example you ran another tool), the script asks you to preview again
    instead of overwriting it.
  - **After sending**, the window shows the updated image as the new original, so
    further previews build on it.
  - **Saved settings:** the window remembers your settings and the CLI path, in
    `astrosharp.conf` in Siril's config folder.
- **From a Siril script (`.ssf`) or the command line**, use `pyscript` with options. Any
  option you leave out uses the saved settings:

  ```
  pyscript AstroSharp.py -model hybrid -psf 3 -aggressiveness 0.66
  pyscript AstroSharp.py -model psf -psf 2.5 -aggressiveness 0.8
  ```

  The options are:
  - `-model hybrid|psf|astroclean`
  - `-psf <1..8, steps of 0.25>`
  - `-aggressiveness <0..1>`
  - `-mode seamless|parity`
  - `-chunksize <n>` (Hybrid in parity mode only)
  - `-executable <path to astrosharp-cli or AstroSharp.app>`

## How it works

1. **Export:** the script takes the current image from Siril (16-bit or 32-bit float) and
   writes it to a temporary 32-bit FITS file, in Siril's row order.
2. **Process:** the AstroSharp CLI runs on that file in the background. Siril's progress bar
   pulses while it works; a 24 MP image takes a few seconds.
3. **Write back:** the result is put back into Siril with an undo point. The image keeps its
   original bit depth, and Siril keeps its own header and colour profile.

## Development

```sh
make test                              # unit tests (creates .venv with numpy and PyQt6)
make test ASTROSHARP_FITS=image.fit    # also the CLI integration test on a real FITS file
```

The tests cover:
- The script's FITS writing/reading and its 16-bit/float conversions.
- The processing flow against a fake Siril, using the real CLI.
- The preview window, off-screen: preview without touching Siril, sending exactly the
  previewed result, processing first when settings changed, refusing to send when Siril's
  image changed, Reset, live preview, field widths, and dragging the divider.
- With `ASTROSHARP_FITS`: a check that processing through the script gives results
  bit-identical to running the CLI directly on the file.

They don't need Siril. Set `ASTROSHARP_CLI` to use a CLI other than the one in
`/Applications`, for example a fresh build in `astrosharp-rs/target/release/bundle/`.

### Manual check in Siril

- [ ] Open a stretched FITS and run **Scripts → AstroSharp**. The image appears; Preview
      shows the processed version, and the image in Siril is unchanged.
- [ ] Zoom, pan, hold Space, and drag the Compare divider.
- [ ] Reset clears the preview and shows Siril's current image again.
- [ ] Send to Siril updates Siril's image, and the console shows a green "applied" line.
      Ctrl/Cmd+Z restores the original; Image → FITS header shows the HISTORY line after
      sending again.
- [ ] Preview, then change the image in Siril (for example with Undo), then Send to Siril.
      The script warns instead of overwriting.
- [ ] Repeat with a 16-bit image (the bit depth is unchanged afterwards).
- [ ] Open a linear stack. The window shows the "looks linear" note, and processing logs a
      salmon note in the console.
- [ ] In a `.ssf` script: `pyscript AstroSharp.py -model psf -psf 2.5` processes the loaded
      image without opening the window.
