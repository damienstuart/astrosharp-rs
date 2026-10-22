# AstroSharp for Siril

A Siril script that runs [AstroSharp](../README.md) on the image currently open in
Siril. It offers AstroSharp's models:
- **Dual PSF** (the default) sharpens the stars and the rest of the image with different
  PSF models, mixed through a feathered star mask.
- **Hybrid**, **PSF** and **AstroClean** sharpen, denoise, or do both.
- **Star Mask** produces the mask itself, for checking it or for use in other tools.

You can preview the result and compare it with the original, using zoom, pan and a
draggable before/after divider like the AstroSharp app. When you're happy, **Send to
Siril** updates the image in place with an undo point, so **Edit → Undo** (Ctrl/Cmd+Z)
restores it. The settings are recorded in the image's FITS HISTORY, for example
`AstroSharp: Dual PSF 3/2 aggr 0.66` (DSO PSF 3, star PSF 2). Star mask settings are added
when they aren't the defaults.

AstroSharp is meant as a **final touch-up on a stretched (non-linear) image**. If the image
looks linear, the script logs a warning in Siril's console, but it still processes the image.

## Requirements

- Siril 1.4 or later. The script uses Siril's bundled Python with numpy and PyQt6.
- AstroSharp 0.5.0 or later for Dual PSF and Star Mask; the other models work from
  0.3.0. Install it from a [release](https://github.com/damienstuart/astrosharp-rs/releases),
  or run `make install` in this repository, which puts `AstroSharp.app`, with its CLI at
  `Contents/MacOS/astrosharp-cli`, in `/Applications`. The script finds it there;
  otherwise choose the CLI, or `AstroSharp.app` itself, with **Browse…**.
  - The window shows the CLI's version, with a note if it's too old for the chosen model.
  - With AstroSharp 0.5.1 or later the progress bars show a percentage; with earlier
    versions they show that it's busy.

## Install

Download `AstroSharp.py` from a [release](https://github.com/damienstuart/astrosharp-rs/releases)
(or take it from this folder) and copy it into a folder on Siril's script path, for example
`~/.siril/scripts`. Siril lists its script folders under **Preferences → Scripts**. From a
checkout of this repository:

```sh
make siril-install                 # from the repository root: copies to ~/.siril/scripts
make install                       # the same, from this folder (SIRIL_SCRIPTS=... to change)
```

Then refresh Siril's script list (**Scripts → Refresh Script List**, or restart Siril).
The script appears in the **Scripts** menu.

## Use

- **Scripts → AstroSharp** opens a window with the settings on the left and the image on
  the right:
  - **Settings:** choose the model, PSF FWHM, Aggressiveness and, for Hybrid, Dual PSF and
    Star Mask, the mode.
    - **Dual PSF** has two PSF sliders: **DSO PSF** for the image as a whole and **Star
      PSF** for the stars.
    - Dual PSF and Star Mask have a **Star mask** box with the R app's five settings:
      Sensitivity, Fix mask, Feathering, Blend and Blend 2. **Defaults** restores them.
      Tune the mask with Star Mask, then switch to Dual PSF.
    - **Star Mask on a colour image** gives a grey mask in all three channels, so the
      image stays RGB.
  - **Process** processes a copy of the image to preview the result; Siril's image is not
    changed. **Live preview** (off by default) refreshes it whenever a setting changes.
  - **Progress:** while it runs, a progress bar under the buttons and Siril's own progress
    bar show how far it has got.
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
  pyscript AstroSharp.py -model dual-psf -psf 3 -starpsf 2 -aggressiveness 0.66
  pyscript AstroSharp.py -model hybrid -psf 3 -aggressiveness 0.66
  pyscript AstroSharp.py -model psf -psf 2.5 -aggressiveness 0.8
  pyscript AstroSharp.py -model star-mask -sensitivity 95 -feathering 3
  ```

  The options are:
  - `-model dual-psf|hybrid|psf|astroclean|star-mask`
  - `-psf <1..8, steps of 0.25>`: for Dual PSF, the PSF used away from the stars
  - `-starpsf <1..8, steps of 0.25>` (Dual PSF)
  - `-aggressiveness <0..1>`
  - `-mode seamless|parity`
  - `-chunksize <n>` (Hybrid and Dual PSF in parity mode only)
  - Star mask (Dual PSF and Star Mask): `-sensitivity <0..100>` (default 98),
    `-fixmask <n>` (2), `-feathering <sigma>` (5), `-blend <a>` (9), `-blend2 <b>` (100)
  - `-executable <path to astrosharp-cli or AstroSharp.app>`

## How it works

1. **Export:** the script takes the current image from Siril (16-bit or 32-bit float) and
   writes it to a temporary 32-bit FITS file, in Siril's row order.
2. **Process:** the AstroSharp CLI runs on that file in the background; a 24 MP image takes
   a few seconds. Its `--progress` output drives the progress bars.
3. **Write back:** the result is put back into Siril with an undo point. The image keeps its
   original bit depth, and Siril keeps its own header and colour profile.

## Development

From the repository root, `make siril-test` builds the CLI and tests the script against
that build:

```sh
make siril-test                              # unit tests (creates siril/.venv with numpy and PyQt6)
make siril-test ASTROSHARP_FITS=image.fit    # also the CLI integration test on a real FITS file
```

Inside this folder, `make test` does the same against the installed AstroSharp (or the CLI
in `ASTROSHARP_CLI`).

The tests cover:
- The script's FITS writing/reading and its 16-bit/float conversions.
- The command line for each model, the HISTORY text and the per-model CLI version check.
- The processing flow against a fake Siril, using the real CLI, for Hybrid, Dual PSF and
  Star Mask (which keeps colour images RGB).
- Progress: reading the CLI's progress lines (with a stand-in CLI, and with the real one
  if it's 0.5.1 or later).
- The preview window, off-screen: preview without touching Siril, sending exactly the
  previewed result, processing first when settings changed, refusing to send when Siril's
  image changed, Reset, live preview, field widths, dragging the divider, the rows each
  model shows, saving and restoring the star mask settings, and the progress bar (a
  percentage, or a busy bar with an older CLI).
- With `ASTROSHARP_FITS`: a check that processing through the script gives results
  bit-identical to running the CLI directly on the file.

They don't need Siril.

### Manual check in Siril

- [ ] Open a stretched FITS and run **Scripts → AstroSharp**. The image appears; Process
      shows the processed version, and the image in Siril is unchanged.
- [ ] On a large image, the progress bar under the buttons and Siril's progress bar both
      advance while it processes, then the window's bar disappears.
- [ ] Zoom, pan, hold Space, and drag the Compare divider.
- [ ] Reset clears the preview and shows Siril's current image again.
- [ ] Send to Siril updates Siril's image, and the console shows a green "applied" line.
      Ctrl/Cmd+Z restores the original; Image → FITS header shows the HISTORY line after
      sending again.
- [ ] Process, then change the image in Siril (for example with Undo), then Send to Siril.
      The script warns instead of overwriting.
- [ ] Repeat with a 16-bit image (the bit depth is unchanged afterwards).
- [ ] Open a linear stack. The window shows the "looks linear" note, and processing logs a
      salmon note in the console.
- [ ] Choose Star Mask and Process: the mask appears. Change Sensitivity or Feathering and
      Process again; then choose Dual PSF with a smaller Star PSF and Send to Siril.
- [ ] In a `.ssf` script: `pyscript AstroSharp.py -model psf -psf 2.5` processes the loaded
      image without opening the window.
