# AstroSharp for Siril
#
# Runs AstroSharp (https://github.com/deepskydetail/AstroSharp, Rust port) on the image
# currently loaded in Siril: sharpen (PSF), denoise (AstroClean), both (Hybrid), sharpen
# stars and the rest of the image with different PSFs (Dual PSF), or make its star mask
# (Star Mask). The image is updated in place with an undo point, so Siril's Undo restores
# it.
#
# AstroSharp is meant as a final touch-up on a stretched (non-linear) image.
#
# Requirements:
#   - Siril 1.4 or later
#   - AstroSharp 0.3.0 or later (0.5.0 or later for Dual PSF and Star Mask; 0.5.1 or later
#     for a progress percentage). Its CLI is found where the installers put it (see
#     default_cli_paths) or on the PATH, or choose it in the window.
#
# Usage:
#   - From Siril's Scripts menu: opens a window with the settings and a preview with a
#     before/after comparison; "Process" makes the preview and "Send to Siril" applies it.
#   - From a Siril script (.ssf) or the command line, e.g.:
#       pyscript AstroSharp.py -model dual-psf -psf 3 -starpsf 2 -aggressiveness 0.66
#     Options not given use the settings last saved from the window.
#
# Version: 0.4.1
# 0.1.0: First version.
# 0.2.0: Preview with zoom/pan and before/after divider; "Send to Siril".
# 0.3.0: Dual PSF and Star Mask models (AstroSharp 0.5.0), with the star mask settings;
#        Dual PSF is the default model.
# 0.4.0: Progress bar while processing (a percentage with AstroSharp 0.5.1); the Preview
#        button is now "Process".
# 0.4.1: Finds the CLI where the Windows and Linux installers put it, or on the PATH.

from __future__ import annotations

import argparse
import hashlib
import json
import ntpath
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import sirilpy as s

s.ensure_installed("PyQt6")

import numpy as np  # noqa: E402 (installed with sirilpy)

VERSION = "0.4.1"
MIN_CLI_VERSION = (0, 3, 0)
CONFIG_NAME = "astrosharp.conf"
MODELS = {"dual-psf": "Dual PSF", "hybrid": "Hybrid", "psf": "PSF",
          "astroclean": "AstroClean", "star-mask": "Star Mask"}
# Models that need a newer CLI than MIN_CLI_VERSION.
MODEL_MIN_VERSION = {"dual-psf": (0, 5, 0), "star-mask": (0, 5, 0)}
MASK_MODELS = ("dual-psf", "star-mask")
# The first CLI that reports progress (--progress).
PROGRESS_VERSION = (0, 5, 1)
PROGRESS_MESSAGE = "AstroSharp: processing…"
MODES = {"seamless": "Seamless", "parity": "R parity"}
FITS_BLOCK = 2880
# Same rule as AstroSharp: 99% of pixels darker than this means the image looks linear.
LINEAR_P99 = 0.15


@dataclass
class Params:
    model: str = "dual-psf"
    psf: float = 3.0
    aggressiveness: float = 0.66
    mode: str = "seamless"
    chunk_size: int = 325
    # Dual PSF: the PSF used on the stars (`psf` is used everywhere else).
    star_psf: float = 3.0
    # Star mask (Dual PSF and Star Mask), with the R app's defaults.
    sensitivity: float = 98.0
    fix_mask: int = 2
    feathering: float = 5.0
    blend: float = 9.0
    blend2: float = 100.0

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        p = cls()
        if d.get("model") in MODELS:
            p.model = d["model"]
        if d.get("mode") in MODES:
            p.mode = d["mode"]
        for key, kind in (("psf", float), ("aggressiveness", float), ("chunk_size", int),
                          ("star_psf", float), ("sensitivity", float), ("fix_mask", int),
                          ("feathering", float), ("blend", float), ("blend2", float)):
            try:
                setattr(p, key, kind(d[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return p

    def uses_mask(self) -> bool:
        return self.model in MASK_MODELS

    def mask_text(self) -> str:
        """The star mask settings, e.g. `98/2/5/9/100` (sensitivity, fix mask, feathering,
        blend, blend 2)."""
        return "/".join(fmt(v) for v in (self.sensitivity, self.fix_mask, self.feathering,
                                         self.blend, self.blend2))

    def default_mask(self) -> bool:
        d = Params()
        return self.mask_text() == d.mask_text()


# --- Helpers (no Siril or Qt needed) -------------------------------------------------


def write_fits(path: str, data: np.ndarray) -> None:
    """Writes `data` (values in [0, 1], shaped (h, w) or (3, h, w) as Siril returns it,
    rows bottom-up) as a minimal 32-bit float FITS file."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2 or (data.ndim == 3 and data.shape[0] == 3):
        dims = data.shape[::-1]  # NAXIS1 = width, NAXIS2 = height, NAXIS3 = channels
    else:
        raise ValueError(f"unsupported image shape {data.shape}")
    cards = [f"SIMPLE  = {'T':>20}", f"BITPIX  = {-32:>20}", f"NAXIS   = {len(dims):>20}"]
    cards += [f"{'NAXIS' + str(i + 1):<8}= {n:>20}" for i, n in enumerate(dims)]
    cards += ["ROWORDER= 'BOTTOM-UP'", "END"]
    header = "".join(c.ljust(80) for c in cards).encode("ascii")
    header += b" " * (-len(header) % FITS_BLOCK)
    body = data.astype(">f4").tobytes()
    body += b"\0" * (-len(body) % FITS_BLOCK)
    with open(path, "wb") as f:
        f.write(header)
        f.write(body)


def read_fits(path: str) -> np.ndarray:
    """Reads the 32-bit float primary image of a FITS file in stored (bottom-up) row
    order, shaped (h, w) or (channels, h, w)."""
    values = {}
    with open(path, "rb") as f:
        while True:
            block = f.read(FITS_BLOCK)
            if len(block) < FITS_BLOCK:
                raise ValueError("truncated FITS header")
            cards = [block[i:i + 80].decode("ascii") for i in range(0, FITS_BLOCK, 80)]
            for card in cards:
                key = card[:8].strip()
                if key == "END":
                    break
                if card[8:10] == "= " and key not in values:
                    values[key] = card[10:].split("/")[0].strip()
            else:
                continue
            break
        if int(values.get("BITPIX", 0)) != -32:
            raise ValueError("expected a 32-bit float FITS image")
        dims = [int(values[f"NAXIS{i}"]) for i in range(1, int(values["NAXIS"]) + 1)]
        count = int(np.prod(dims))
        data = np.frombuffer(f.read(count * 4), dtype=">f4")
    if data.size != count:
        raise ValueError("truncated FITS data")
    data = data.reshape(dims[::-1]).astype(np.float32)
    if values.get("ROWORDER", "").strip("' ") == "TOP-DOWN":
        data = np.flip(data, axis=-2)
    return data


def to_float(pixels: np.ndarray) -> np.ndarray:
    """Siril pixel data (uint16 or float32) as float32 in [0, 1]."""
    if pixels.dtype == np.uint16:
        return pixels.astype(np.float32) / np.float32(65535.0)
    if pixels.dtype == np.float32:
        return pixels
    raise ValueError(f"unsupported pixel type {pixels.dtype}")


def from_float(result: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Converts processed float data back to the image's original sample type."""
    clipped = np.clip(result, 0.0, 1.0)
    if dtype == np.uint16:
        return np.rint(clipped.astype(np.float64) * 65535.0).astype(np.uint16)
    return clipped.astype(np.float32)


def display_u8(pixels: np.ndarray) -> np.ndarray:
    """8-bit display copy of Siril pixel data: top row first, (h, w) or (h, w, 3)."""
    data = to_float(pixels)
    if data.ndim == 3:
        data = data[0] if data.shape[0] == 1 else np.moveaxis(data, 0, -1)
    u8 = np.rint(np.clip(data, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(u8[::-1])  # Siril stores rows bottom-up


def fingerprint(pixels: np.ndarray) -> str:
    """Cheap identity of an image (shape, type and a sample of pixels), used to notice
    when Siril's image changed after a preview."""
    flat = pixels.reshape(-1)
    sample = flat[:: max(1, flat.size // 2_000_000)]
    h = hashlib.sha1(f"{pixels.shape}{pixels.dtype}".encode())
    h.update(np.ascontiguousarray(sample).tobytes())
    return h.hexdigest()


def looks_linear(pixels: np.ndarray) -> bool:
    """Whether the image looks unstretched (see LINEAR_P99)."""
    data = to_float(pixels)
    lum = data.mean(axis=0) if data.ndim == 3 else data
    flat = lum.reshape(-1)
    sample = flat[:: max(1, flat.size // 1_000_000)]
    return bool(sample.size) and float(np.percentile(sample, 99)) < LINEAR_P99


def fmt(x: float) -> str:
    return f"{x:g}"


def fit_shape(result: np.ndarray, shape: tuple) -> np.ndarray:
    """The CLI's output in the shape of Siril's image. A star mask of a colour image is a
    single grey channel; it is repeated in each colour channel so the image stays RGB."""
    if result.ndim == 2 and len(shape) == 3 and shape[0] == 3:
        return np.repeat(result[np.newaxis], 3, axis=0)
    return result.reshape(shape)


def build_command(exe: str, inp: str, out: str, p: Params,
                  progress: bool = False) -> list[str]:
    """The CLI command. Options a model doesn't use are left out, so older CLIs still run
    the models they have. With `progress`, the CLI reports progress on stdout (0.5.1 or
    later)."""
    cmd = [exe, inp, out, "--model", p.model, "--psf", fmt(p.psf),
           "--aggressiveness", fmt(p.aggressiveness), "--mode", p.mode, "--depth", "32"]
    if p.model == "dual-psf":
        cmd += ["--star-psf", fmt(p.star_psf)]
    if p.uses_mask():
        cmd += ["--sensitivity", fmt(p.sensitivity), "--fix-mask", str(p.fix_mask),
                "--feathering", fmt(p.feathering), "--blend", fmt(p.blend),
                "--blend2", fmt(p.blend2)]
    if p.model in ("hybrid", "dual-psf") and p.mode == "parity":
        cmd += ["--chunk-size", str(p.chunk_size)]
    if progress:
        cmd.append("--progress")
    return cmd


def parse_progress(line: str) -> Optional[float]:
    """The fraction in a `progress <fraction>` line from the CLI, or None."""
    if line.startswith("progress "):
        try:
            return min(max(float(line[9:]), 0.0), 1.0)
        except ValueError:
            pass
    return None


def settings_text(p: Params) -> str:
    """Short description of the settings, e.g. `Hybrid PSF 3 aggr 0.66`,
    `Dual PSF 3/2 aggr 0.66` (DSO/star PSF) or `Star Mask 98/2/5/9/100`. Dual PSF lists
    the star mask settings when they aren't the defaults."""
    parts = [MODELS[p.model]]
    if p.model == "dual-psf":
        parts.append(f"{fmt(p.psf)}/{fmt(p.star_psf)}")
    elif p.model == "star-mask":
        parts.append(p.mask_text())
    elif p.model != "astroclean":
        parts.append(f"PSF {fmt(p.psf)}")
    if p.model != "star-mask":
        parts.append(f"aggr {p.aggressiveness:.2f}")
    if p.mode == "parity":
        if p.model in ("hybrid", "dual-psf"):
            parts.append(f"parity chunk {p.chunk_size}")
        elif p.model == "star-mask":
            parts.append("parity")
    if p.model == "dual-psf" and not p.default_mask():
        parts.append(f"mask {p.mask_text()}")
    return " ".join(parts)


def history_text(p: Params) -> str:
    """Undo/HISTORY entry (Siril keeps at most 70 bytes)."""
    return f"AstroSharp: {settings_text(p)}"[:70]


def default_cli_paths(platform: Optional[str] = None) -> list[str]:
    """Where the AstroSharp installers put the CLI on `platform` (default: this one), then
    any astrosharp-cli on the PATH."""
    platform = platform or sys.platform
    if platform == "darwin":
        paths = ["/Applications/AstroSharp.app/Contents/MacOS/astrosharp-cli"]
    elif platform.startswith("win"):
        # The installer's per-user location, %LOCALAPPDATA%\Programs\AstroSharp.
        local = os.environ.get("LOCALAPPDATA") or ntpath.join(
            os.path.expanduser("~"), "AppData", "Local")
        paths = [ntpath.join(local, "Programs", "AstroSharp", "astrosharp-cli.exe")]
    else:
        # tools/install_linux.sh puts it in ~/.local/bin.
        paths = [os.path.join(os.path.expanduser("~"), ".local", "bin", "astrosharp-cli")]
    on_path = shutil.which("astrosharp-cli")
    if on_path and on_path not in paths:
        paths.append(on_path)
    return paths


# Where this platform's installer puts the CLI.
DEFAULT_CLI = default_cli_paths()[0]


def resolve_executable(path: str) -> str:
    """Accepts the CLI itself or the AstroSharp.app bundle containing it."""
    if path.endswith(".app"):
        return os.path.join(path, "Contents", "MacOS", "astrosharp-cli")
    return path


def find_executable(configured: Optional[str]) -> Optional[str]:
    """The configured CLI if it exists, else the first of default_cli_paths() that does."""
    for candidate in [configured, *default_cli_paths()]:
        if candidate:
            candidate = resolve_executable(candidate)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def min_version(model: str) -> tuple[int, int, int]:
    """The oldest AstroSharp CLI that has `model`."""
    return MODEL_MIN_VERSION.get(model, MIN_CLI_VERSION)


def version_text(version: tuple[int, int, int]) -> str:
    return ".".join(map(str, version))


def has_progress(version: Optional[tuple[int, int, int]]) -> bool:
    """Whether a CLI of this version can report progress."""
    return version is not None and version >= PROGRESS_VERSION


def cli_version(exe: str) -> Optional[tuple[int, int, int]]:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(g) for g in m.groups()) if m else None


def check_executable(configured: Optional[str], model: Optional[str] = None) -> str:
    """The usable CLI path (for `model`, if given), or raises RuntimeError explaining what
    is wrong."""
    exe = find_executable(configured)
    if exe is None:
        raise RuntimeError(
            f"AstroSharp CLI not found. Install AstroSharp (its CLI goes in {DEFAULT_CLI}) or "
            f"choose the CLI in the AstroSharp window.")
    version = cli_version(exe)
    if version is None or version < MIN_CLI_VERSION:
        raise RuntimeError(f"{exe} is not AstroSharp {version_text(MIN_CLI_VERSION)} or later.")
    if model is not None and version < min_version(model):
        raise RuntimeError(f"{MODELS[model]} needs AstroSharp {version_text(min_version(model))} "
                           f"or later; {exe} is {version_text(version)}.")
    return exe


# --- Siril integration ----------------------------------------------------------------


def load_config(siril) -> dict:
    try:
        with open(os.path.join(siril.get_siril_configdir(), CONFIG_NAME)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(siril, executable: str, p: Params) -> None:
    try:
        with open(os.path.join(siril.get_siril_configdir(), CONFIG_NAME), "w") as f:
            json.dump({"executable": executable, **asdict(p)}, f, indent=2)
    except OSError as e:
        siril.log(f"AstroSharp: could not save settings: {e}", s.LogColor.SALMON)


class Job:
    """One AstroSharp run on a copy of Siril's image (pixels as returned by
    get_image_pixeldata). `run` does all the heavy work and is safe in a worker thread;
    `apply` (main thread) puts the result into Siril."""

    def __init__(self, pixels: np.ndarray, exe: str, params: Params, progress: bool = False):
        self.pixels, self.exe, self.params = pixels, exe, params
        # Whether the CLI reports progress (see has_progress).
        self.progress = progress
        self.code: Optional[int] = None
        self.stderr = ""
        self.result: Optional[np.ndarray] = None
        self.seconds = 0.0

    @classmethod
    def from_siril(cls, siril, exe: str, params: Params, progress: bool = False) -> Job:
        if not siril.is_image_loaded():
            raise RuntimeError("No image is loaded in Siril.")
        return cls(siril.get_image_pixeldata(), exe, params, progress)

    def run(self, on_progress: Optional[Callable[[float], None]] = None) -> None:
        """Runs the CLI on a temporary FITS copy and reads the result back. If the CLI
        reports progress, `on_progress` gets each fraction (0 to 1) as it arrives."""
        started = time.monotonic()
        data = to_float(self.pixels)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        with tempfile.TemporaryDirectory(prefix="astrosharp-") as tmp:
            inp, out = os.path.join(tmp, "in.fit"), os.path.join(tmp, "out.fit")
            try:
                write_fits(inp, data)
                self.code, self.stderr = self.run_cli(inp, out, tmp, on_progress)
                if self.code == 0:
                    self.result = from_float(fit_shape(read_fits(out), self.pixels.shape),
                                             self.pixels.dtype)
            except (OSError, ValueError) as e:
                self.code, self.stderr = -1, str(e)
        self.seconds = time.monotonic() - started

    def run_cli(self, inp: str, out: str, tmp: str,
                on_progress: Optional[Callable[[float], None]]) -> tuple[int, str]:
        """Runs the CLI, passing its progress lines to `on_progress` (if it was asked to
        report progress). Returns its exit code and stderr (kept in a file, so a full pipe
        can never stall it)."""
        cmd = build_command(self.exe, inp, out, self.params, self.progress)
        if not self.progress:
            on_progress = None
        with open(os.path.join(tmp, "stderr.txt"), "w+") as err:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True) as proc:
                for line in proc.stdout:
                    fraction = parse_progress(line)
                    if fraction is not None and on_progress is not None:
                        try:
                            on_progress(fraction)
                        except Exception:  # progress is only shown; never stop the run
                            pass
            err.seek(0)
            return proc.returncode, err.read()

    def notes(self) -> list[str]:
        return [line for line in self.stderr.splitlines() if line.startswith("note:")]

    def log_notes(self, siril) -> None:
        for note in self.notes():
            siril.log(f"AstroSharp {note}", s.LogColor.SALMON)

    def check(self) -> np.ndarray:
        """The processed pixels, or raises RuntimeError with AstroSharp's error."""
        if self.result is None:
            detail = self.stderr.strip() or f"exit code {self.code}"
            raise RuntimeError(f"AstroSharp failed: {detail}")
        return self.result

    def apply(self, siril) -> None:
        """Replaces Siril's image with the result, with an undo point."""
        result = self.check()
        with siril.image_lock():
            siril.undo_save_state(history_text(self.params))
            siril.set_image_pixeldata(result)
        siril.log(f"{history_text(self.params)} applied", s.LogColor.GREEN)


def connect():
    siril = s.SirilInterface()
    try:
        siril.connect()
    except s.SirilConnectionError:
        print("Error: failed to connect to Siril", file=sys.stderr)
        sys.exit(1)
    try:
        siril.cmd("requires", "1.4.0")
    except s.CommandError:
        sys.exit(1)
    return siril


def run_cli_mode(args) -> None:
    siril = connect()
    config = load_config(siril)
    p = Params.from_dict(config)
    for key, value in (("model", args.model), ("psf", args.psf),
                       ("aggressiveness", args.aggressiveness), ("mode", args.mode),
                       ("chunk_size", args.chunksize), ("star_psf", args.starpsf),
                       ("sensitivity", args.sensitivity), ("fix_mask", args.fixmask),
                       ("feathering", args.feathering), ("blend", args.blend),
                       ("blend2", args.blend2)):
        if value is not None:
            setattr(p, key, value)
    try:
        exe = check_executable(args.executable or config.get("executable"), p.model)
        if args.executable:
            save_config(siril, exe, Params.from_dict(config))
        job = Job.from_siril(siril, exe, p, has_progress(cli_version(exe)))
        siril.update_progress(PROGRESS_MESSAGE, 0.0 if job.progress else -1.0)
        try:
            job.run(on_progress=lambda f: siril.update_progress(PROGRESS_MESSAGE, f))
        finally:
            siril.reset_progress()
        job.log_notes(siril)
        job.apply(siril)
    except Exception as e:  # report every failure in Siril's log
        siril.log(str(e), s.LogColor.RED)
        sys.exit(1)


def create_window(siril):
    """Builds the settings and preview window (PyQt6 is imported here so the helpers work
    without it)."""
    from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF
    from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
                                 QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                                 QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout,
                                 QWidget)

    accent = QColor(255, 200, 40)

    def to_qimage(u8: np.ndarray) -> QImage:
        h, w = u8.shape[:2]
        if u8.ndim == 3:
            img = QImage(u8.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        else:
            img = QImage(u8.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        return img.copy()  # own the pixels

    class ValueSlider(QWidget):
        """A slider for a decimal setting, with its value shown alongside."""

        valueChanged = pyqtSignal(float)

        def __init__(self, minimum: float, maximum: float, step: float, page: float):
            super().__init__()
            self.step = step
            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setRange(round(minimum / step), round(maximum / step))
            self.slider.setPageStep(round(page / step))
            self.label = QLabel()
            self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            widest = self.label.fontMetrics().horizontalAdvance(f"{maximum:.2f}")
            self.label.setFixedWidth(widest + 8)
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self.slider, 1)
            row.addWidget(self.label)
            self.slider.valueChanged.connect(self._changed)
            self._changed()

        def value(self) -> float:
            return round(self.slider.value() * self.step, 6)

        def setValue(self, value: float):
            self.slider.setValue(round(value / self.step))

        def _changed(self, *_):
            self.label.setText(f"{self.value():.2f}")
            self.valueChanged.emit(self.value())

    class Worker(QThread):
        done = pyqtSignal()
        progress = pyqtSignal(float)

        def __init__(self, job: Job):
            super().__init__()
            self.job = job
            self.display: Optional[np.ndarray] = None

        def run(self):
            self.job.run(on_progress=self.progress.emit)
            if self.job.result is not None:
                self.display = display_u8(self.job.result)
            self.done.emit()

    class ImageView(QWidget):
        """Zoomable/pannable view of the original with the processed image on top, and
        an optional before/after divider (original on the left)."""

        handle = 14.0
        reduced_side = 2048  # zoomed out, draw a reduced copy (fast for large images)

        def __init__(self):
            super().__init__()
            self.setMinimumSize(480, 360)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setMouseTracking(True)
            self.before = self.after = None  # [full, reduced?] QImages
            self.compare = False
            self.split = 0.5
            self.show_original = False
            self.zoom, self.center = 1.0, QPointF()
            self.fitted = True
            self.drag, self.last = None, QPointF()

        @classmethod
        def levels(cls, img: Optional[QImage]):
            if img is None:
                return None
            levels = [img]
            if max(img.width(), img.height()) > cls.reduced_side:
                levels.append(img.scaled(cls.reduced_side, cls.reduced_side,
                                         Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
            return levels

        def set_before(self, img: Optional[QImage]):
            old = self.before[0].size() if self.before else None
            self.before = self.levels(img)
            if img is not None and img.size() != old:
                self.fit()
            self.update()

        def set_after(self, img: Optional[QImage]):
            self.after = self.levels(img)
            self.update()

        def set_compare(self, on: bool):
            self.compare = on
            self.update()

        def fit(self):
            self.fitted = True
            if self.before:
                w, h = self.before[0].width(), self.before[0].height()
                self.zoom = max(min(self.width() / w, self.height() / h), 1e-3)
                self.center = QPointF(w / 2, h / 2)
            self.update()

        def to_screen(self, p: QPointF) -> QPointF:
            return QPointF(self.width() / 2, self.height() / 2) + (p - self.center) * self.zoom

        def to_image(self, p: QPointF) -> QPointF:
            return self.center + (p - QPointF(self.width() / 2, self.height() / 2)) / self.zoom

        def split_x(self) -> float:
            return self.split * self.width()

        def near_handle(self, x: float) -> bool:
            return self.compare and self.after is not None and abs(x - self.split_x()) <= self.handle

        def draw(self, painter: QPainter, levels):
            full = levels[0]
            w, h = full.width(), full.height()
            target = QRectF(self.to_screen(QPointF(0, 0)), self.to_screen(QPointF(w, h)))
            if len(levels) > 1 and target.width() <= levels[1].width():
                small = levels[1]
                painter.drawImage(target, small, QRectF(0, 0, small.width(), small.height()))
            else:
                # Full resolution: only the visible part, so zoomed-in drawing stays cheap.
                visible = target.intersected(QRectF(self.rect()))
                if not visible.isEmpty():
                    src = QRectF(self.to_image(visible.topLeft()),
                                 self.to_image(visible.bottomRight()))
                    painter.drawImage(visible, full, src)

        def paintEvent(self, event):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(24, 24, 24))
            if not self.before:
                p.setPen(QColor(150, 150, 150))
                p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded in Siril")
                return
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
            self.draw(p, self.before)
            if self.after and not self.show_original:
                x = self.split_x() if self.compare else 0.0
                p.save()
                p.setClipRect(QRectF(x, 0, self.width() - x, self.height()))
                self.draw(p, self.after)
                p.restore()
                if self.compare:
                    self.draw_divider(p, x)
            p.setPen(QColor(255, 255, 255, 170))
            p.setFont(QFont("Menlo", 10))
            p.drawText(QRectF(8, self.height() - 22, 300, 18), Qt.AlignmentFlag.AlignLeft,
                       f"{self.zoom * 100:.0f}%")

        def draw_divider(self, p: QPainter, x: float):
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(QPen(QColor(255, 255, 255, 180), 1.5))
            p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
            p.setFont(QFont("Helvetica", 11))
            p.drawText(QRectF(x - 208, 6, 200, 18), Qt.AlignmentFlag.AlignRight, "original")
            p.drawText(QRectF(x + 8, 6, 200, 18), Qt.AlignmentFlag.AlignLeft, "processed")
            c = QPointF(x, self.height() / 2)
            p.setPen(QPen(QColor(0, 0, 0, 140), 1.5))
            p.setBrush(accent)
            p.drawEllipse(c, self.handle, self.handle)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(43, 34, 0))
            for d in (-1.0, 1.0):
                tip, base = c + QPointF(9 * d, 0), c + QPointF(3 * d, 0)
                p.drawPolygon(QPolygonF([tip, base + QPointF(0, -5.5), base + QPointF(0, 5.5)]))

        def mousePressEvent(self, e):
            self.setFocus()
            if e.button() == Qt.MouseButton.LeftButton:
                self.drag = "split" if self.near_handle(e.position().x()) else "pan"
                self.last = e.position()

        def mouseMoveEvent(self, e):
            pos = e.position()
            if self.drag == "split":
                self.split = min(max(pos.x() / max(self.width(), 1), 0.0), 1.0)
                self.update()
            elif self.drag == "pan":
                self.center -= (pos - self.last) / self.zoom
                self.last, self.fitted = pos, False
                self.update()
            self.setCursor(Qt.CursorShape.SplitHCursor if self.drag == "split" or
                           self.near_handle(pos.x()) else Qt.CursorShape.ArrowCursor)

        def mouseReleaseEvent(self, e):
            self.drag = None

        def mouseDoubleClickEvent(self, e):
            self.fit()

        def wheelEvent(self, e):
            if not self.before:
                return
            factor = 1.0015 ** e.angleDelta().y()
            anchor = self.to_image(e.position())
            self.zoom = min(max(self.zoom * factor, 0.02), 32.0)
            self.center = anchor - (e.position() - QPointF(self.width() / 2,
                                                           self.height() / 2)) / self.zoom
            self.fitted = False
            self.update()

        def keyPressEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.show_original = True
                self.update()
            else:
                super().keyPressEvent(e)

        def keyReleaseEvent(self, e):
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.show_original = False
                self.update()
            else:
                super().keyReleaseEvent(e)

        def resizeEvent(self, e):
            if self.fitted:
                self.fit()

    class Window(QWidget):
        def __init__(self, siril):
            super().__init__()
            self.siril = siril
            self.worker: Optional[Worker] = None
            self.preview: Optional[Job] = None      # last successful preview
            self.preview_fp: Optional[str] = None   # fingerprint of the image it used
            self.before_fp: Optional[str] = None    # image shown as "original"
            self.send_when_done = False
            self.rerun = False
            self.cli: tuple = (None, None)  # (CLI path, its version), for the version label
            config = load_config(siril)
            p = Params.from_dict(config)
            self.setWindowTitle(f"AstroSharp – v{VERSION}")

            self.model = QComboBox()
            for key, name in MODELS.items():
                self.model.addItem(name, key)
            self.model.setCurrentIndex(list(MODELS).index(p.model))
            self.psf = ValueSlider(1.0, 8.0, step=0.25, page=1.0)
            self.psf.setValue(p.psf)
            self.psf.setToolTip("70% of the PSF in pixels: measure the FWHM of a small, "
                                "unsaturated star and divide by 2.35. For Dual PSF, the "
                                "model used away from the stars (the deep-sky object).")
            self.star_psf = ValueSlider(1.0, 8.0, step=0.25, page=1.0)
            self.star_psf.setValue(p.star_psf)
            self.star_psf.setToolTip("Dual PSF: the PSF model used on the stars, where the "
                                     "star mask is set.")
            self.aggr = ValueSlider(0.01, 1.0, step=0.01, page=0.1)
            self.aggr.setValue(p.aggressiveness)
            self.aggr.setToolTip("Blend between the model output (1) and the input (0).")
            self.mode = QComboBox()
            for key, name in MODES.items():
                self.mode.addItem(name, key)
            self.mode.setCurrentIndex(list(MODES).index(p.mode))
            self.mode.setToolTip("Seamless: whole-image processing at full precision, with "
                                 "one star mask for the whole image. R parity: reproduces "
                                 "the original R app exactly.")
            self.chunk = QSpinBox(minimum=50, maximum=750)
            self.chunk.setValue(p.chunk_size)
            self.chunk.setMinimumWidth(90)
            self.form = QFormLayout()
            # The macOS default keeps fields at their size hint, which clips the numbers.
            self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.form.addRow("Model", self.model)
            self.form.addRow("PSF FWHM (px)", self.psf)
            self.form.addRow("Star PSF FWHM (px)", self.star_psf)
            self.form.addRow("Aggressiveness", self.aggr)
            self.form.addRow("Mode", self.mode)
            self.form.addRow("Chunk size", self.chunk)

            # Star mask (Dual PSF and Star Mask), with the R app's names.
            self.sensitivity = QDoubleSpinBox(minimum=1.0, maximum=100.0, decimals=1,
                                              singleStep=0.5)
            self.sensitivity.setToolTip("Percentile of the star response above which pixels "
                                        "count as stars: higher finds fewer stars.")
            self.fix_mask = QSpinBox(minimum=0, maximum=1000)
            self.fix_mask.setToolTip("Leaves out the first regions found scanning down each "
                                     "column from the left, up to this number. Region 0 is "
                                     "normally the background; raise it to drop artefacts "
                                     "at the left edge.")
            self.feathering = QDoubleSpinBox(minimum=0.0, maximum=30.0, decimals=2,
                                             singleStep=0.5)
            self.feathering.setToolTip("Blur (sigma, px) of the mask edges; below 0.1 there "
                                       "is none.")
            self.blend = QDoubleSpinBox(minimum=0.01, maximum=10000.0, decimals=2,
                                        singleStep=0.05)
            self.blend2 = QDoubleSpinBox(minimum=0.01, maximum=10000.0, decimals=2,
                                         singleStep=0.05)
            for w in (self.blend, self.blend2):
                w.setToolTip("Shape of the beta curve that turns the feathered mask into "
                             "weights: it rises around Blend / (Blend + Blend 2), more "
                             "steeply the larger they are.")
            for w in (self.sensitivity, self.fix_mask, self.feathering, self.blend,
                      self.blend2):
                w.setMinimumWidth(90)
            self.set_mask(p)
            mask_defaults = QPushButton("Defaults")
            mask_defaults.setToolTip("Restore the R app's star mask settings.")
            mask_defaults.clicked.connect(lambda: self.set_mask(Params()))
            mask_form = QFormLayout()
            mask_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            mask_form.addRow("Sensitivity", self.sensitivity)
            mask_form.addRow("Fix mask", self.fix_mask)
            mask_form.addRow("Feathering", self.feathering)
            mask_form.addRow("Blend", self.blend)
            mask_form.addRow("Blend 2", self.blend2)
            mask_form.addRow("", mask_defaults)
            self.mask_box = QGroupBox("Star mask")
            self.mask_box.setLayout(mask_form)

            self.live = QCheckBox("Live preview")
            self.live.setToolTip("Update the preview automatically when a setting changes.")
            self.compare = QCheckBox("Compare before/after")
            self.compare.setToolTip("Split the view; drag the divider to move it.")
            help_text = QLabel("Scroll to zoom, drag to pan, double-click to fit. "
                               "Hold Space to see the original.")
            help_text.setWordWrap(True)
            help_text.setStyleSheet("color: gray; font-size: 11px;")

            self.process_btn = QPushButton("Process")
            self.process_btn.setToolTip("Process a copy of the image to preview the result; "
                                        "Siril's image isn't changed.")
            self.process_btn.clicked.connect(lambda: self.start_preview())
            self.reset_btn = QPushButton("Reset")
            self.reset_btn.setToolTip("Discard the preview and reload the image from Siril.")
            self.reset_btn.clicked.connect(lambda: self.reset())
            self.send_btn = QPushButton("Send to Siril")
            self.send_btn.setDefault(True)
            self.send_btn.setToolTip("Apply the previewed result to Siril's image "
                                     "(with an undo point).")
            self.send_btn.clicked.connect(lambda: self.send())
            actions = QHBoxLayout()
            actions.addWidget(self.process_btn)
            actions.addWidget(self.reset_btn)
            actions.addWidget(self.send_btn)

            self.progress = QProgressBar()
            self.progress.setRange(0, 1000)
            self.progress.setFormat("%p%")
            self.progress.hide()
            self.status = QLabel("Press Process to see the result.")
            self.status.setWordWrap(True)
            self.linear_note = QLabel("This image looks linear (unstretched). AstroSharp is "
                                      "meant as a final touch-up on a stretched image.")
            self.linear_note.setWordWrap(True)
            self.linear_note.setStyleSheet(f"color: {accent.name()};")
            self.linear_note.hide()

            self.exe = QLineEdit(config.get("executable") or find_executable(None) or "")
            browse = QPushButton("Browse…")
            browse.clicked.connect(self.browse)
            self.version = QLabel()
            exe_row = QHBoxLayout()
            exe_row.addWidget(self.exe)
            exe_row.addWidget(browse)
            close = QPushButton("Close")
            close.clicked.connect(self.close)

            panel = QVBoxLayout()
            panel.addLayout(self.form)
            panel.addWidget(self.mask_box)
            panel.addWidget(self.live)
            panel.addSpacing(8)
            panel.addWidget(self.compare)
            panel.addWidget(help_text)
            panel.addSpacing(8)
            panel.addLayout(actions)
            panel.addWidget(self.progress)
            panel.addWidget(self.status)
            panel.addWidget(self.linear_note)
            panel.addStretch()
            panel.addWidget(QLabel("AstroSharp CLI"))
            panel.addLayout(exe_row)
            panel.addWidget(self.version)
            panel.addSpacing(8)
            panel.addWidget(close)
            side = QWidget()
            side.setLayout(panel)
            side.setFixedWidth(340)

            self.view = ImageView()
            layout = QHBoxLayout(self)
            layout.addWidget(side)
            layout.addWidget(self.view, 1)
            self.resize(1280, 820)

            self.timer = QTimer(self, singleShot=True, interval=400)
            self.timer.timeout.connect(self.start_preview)
            for w in (self.model, self.mode):
                w.currentIndexChanged.connect(self.settings_changed)
            for w in (self.psf, self.star_psf, self.aggr, self.chunk, self.sensitivity,
                      self.fix_mask, self.feathering, self.blend, self.blend2):
                w.valueChanged.connect(self.settings_changed)
            self.compare.toggled.connect(self.view.set_compare)
            self.exe.editingFinished.connect(self.update_version)
            self.update_rows()  # also shows the CLI version
            self.refresh_original()
            self.update_buttons()

        # --- state ---

        def params(self) -> Params:
            return Params(self.model.currentData(), self.psf.value(), self.aggr.value(),
                          self.mode.currentData(), self.chunk.value(), self.star_psf.value(),
                          round(self.sensitivity.value(), 6), self.fix_mask.value(),
                          round(self.feathering.value(), 6), round(self.blend.value(), 6),
                          round(self.blend2.value(), 6))

        def set_mask(self, p: Params):
            """Shows `p`'s star mask settings."""
            self.sensitivity.setValue(p.sensitivity)
            self.fix_mask.setValue(p.fix_mask)
            self.feathering.setValue(p.feathering)
            self.blend.setValue(p.blend)
            self.blend2.setValue(p.blend2)

        def preview_current(self) -> bool:
            return self.preview is not None and self.preview.params == self.params()

        def running(self) -> bool:
            return self.worker is not None and self.worker.isRunning()

        def update_rows(self):
            model = self.model.currentData()
            dual = model == "dual-psf"
            self.psf.setEnabled(model not in ("astroclean", "star-mask"))
            self.form.labelForField(self.psf).setText(
                "DSO PSF FWHM (px)" if dual else "PSF FWHM (px)")
            self.form.setRowVisible(self.star_psf, dual)
            self.aggr.setEnabled(model != "star-mask")
            self.form.setRowVisible(self.mode, model in ("hybrid", "dual-psf", "star-mask"))
            self.form.setRowVisible(self.chunk, model in ("hybrid", "dual-psf")
                                    and self.mode.currentData() == "parity")
            self.mask_box.setVisible(model in MASK_MODELS)
            self.update_version()

        def update_buttons(self):
            loaded = self.view.before is not None
            idle = not self.running()
            self.process_btn.setEnabled(loaded and idle)
            self.send_btn.setEnabled(loaded and idle)
            self.reset_btn.setEnabled(idle)

        def settings_changed(self):
            self.update_rows()
            if self.preview is not None and not self.preview_current():
                self.status.setText("Settings changed since the preview.")
            if self.live.isChecked():
                self.timer.start()

        def show_original(self, pixels: np.ndarray):
            """Shows `pixels` (Siril's current image) as the original, if it changed."""
            fp = fingerprint(pixels)
            if fp != self.before_fp:
                self.before_fp = fp
                self.view.set_before(to_qimage(display_u8(pixels)))
                self.view.set_after(None)
                self.preview = None
                self.linear_note.setVisible(looks_linear(pixels))

        def refresh_original(self):
            try:
                if self.siril.is_image_loaded():
                    self.show_original(self.siril.get_image_pixeldata())
                    return
            except Exception as e:
                self.siril.log(f"AstroSharp: {e}", s.LogColor.RED)
            self.view.set_before(None)
            self.status.setText("Load an image in Siril, then reopen this window.")

        # --- actions ---

        def reset(self):
            """Discards the preview and shows Siril's current image again."""
            if self.running():
                return
            self.timer.stop()
            self.preview = self.preview_fp = self.before_fp = None
            self.send_when_done = self.rerun = False
            self.refresh_original()
            self.view.set_after(None)
            if self.view.before is not None:
                self.status.setText("Showing Siril's current image. Press Process to see the "
                                    "result.")
            self.update_buttons()

        def start_preview(self, send: bool = False):
            if self.running():
                self.rerun = self.rerun or not send
                self.send_when_done = self.send_when_done or send
                return
            p = self.params()
            try:
                exe = check_executable(self.exe.text(), p.model)
                save_config(self.siril, exe, p)
                job = Job.from_siril(self.siril, exe, p, has_progress(cli_version(exe)))
            except Exception as e:
                self.fail(str(e))
                return
            self.show_original(job.pixels)
            self.send_when_done = send
            self.status.setText("Processing…")
            # A percentage if the CLI reports progress, otherwise a busy bar.
            self.progress.setRange(0, 1000 if job.progress else 0)
            self.progress.setValue(0)
            self.progress.show()
            self.siril.update_progress(PROGRESS_MESSAGE, 0.0 if job.progress else -1.0)
            self.worker = Worker(job)
            # Bound methods of this (main-thread) widget, so Qt delivers the signals on the
            # GUI thread.
            self.worker.progress.connect(self.show_progress)
            self.worker.done.connect(self.preview_done)
            self.worker.start()
            self.update_buttons()

        def show_progress(self, fraction: float):
            self.progress.setValue(round(fraction * 1000))
            self.siril.update_progress(PROGRESS_MESSAGE, fraction)

        def preview_done(self):
            worker = self.worker
            job = worker.job
            worker.wait()
            self.progress.hide()
            self.siril.reset_progress()
            self.update_buttons()
            job.log_notes(self.siril)
            if job.result is None:
                self.send_when_done = self.rerun = False
                self.fail(f"AstroSharp failed: {job.stderr.strip() or f'exit code {job.code}'}")
                return
            self.preview, self.preview_fp = job, fingerprint(job.pixels)
            self.view.set_after(to_qimage(worker.display))
            self.status.setText(f"Preview: {settings_text(job.params)} ({job.seconds:.1f} s)")
            if self.send_when_done and self.preview_current():
                self.send_when_done = False
                self.send()
            elif self.rerun or self.send_when_done or (self.live.isChecked()
                                                       and not self.preview_current()):
                send, self.rerun = self.send_when_done, False
                self.start_preview(send=send)

        def send(self):
            if self.running():
                self.send_when_done = True
                return
            if not self.preview_current():
                self.start_preview(send=True)  # process first, then send
                return
            try:
                current = self.siril.get_image_pixeldata()
            except Exception as e:
                self.fail(str(e))
                return
            if fingerprint(current) != self.preview_fp:
                self.show_original(current)
                message = ("The image in Siril changed since the preview. Press Process "
                           "again (or Send to Siril to process and send).")
                self.status.setText(message)
                self.siril.warning_messagebox(message)
                return
            job = self.preview
            try:
                job.apply(self.siril)
            except s.ProcessingThreadBusyError:
                self.fail("Siril is busy with another task; try again when it has finished.")
                return
            except s.ImageDialogOpenError:
                self.fail("Close Siril's open processing dialog and try again.")
                return
            except Exception as e:
                self.fail(str(e))
                return
            # Siril's image is now the processed one: further previews start from it.
            self.show_original(job.result)
            self.status.setText(f"Sent to Siril: {settings_text(job.params)}. "
                                "Use Siril's Undo to revert.")

        # --- CLI path ---

        def update_version(self):
            exe = find_executable(self.exe.text())
            if exe != self.cli[0]:
                self.cli = (exe, cli_version(exe) if exe else None)
            version = self.cli[1]
            if version is None:
                self.version.setText("CLI not found")
                return
            model = self.model.currentData()
            need = min_version(model)
            if version < MIN_CLI_VERSION:
                note = f" (too old; {version_text(MIN_CLI_VERSION)} or later needed)"
            elif version < need:
                note = f" (too old for {MODELS[model]}; {version_text(need)} or later needed)"
            else:
                note = ""
            self.version.setText(f"AstroSharp {version_text(version)}{note}")

        def browse(self):
            mac = sys.platform == "darwin"
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose the AstroSharp CLI" + (" (or AstroSharp.app)" if mac else ""),
                os.path.dirname(self.exe.text())
                or ("/Applications" if mac else os.path.dirname(DEFAULT_CLI)))
            if path:
                self.exe.setText(resolve_executable(path))
                self.cli = (None, None)  # check again, even if it's the same path
                self.update_version()

        def closeEvent(self, event):
            # Never destroy a running worker thread.
            self.timer.stop()
            if self.running():
                self.worker.wait()
            super().closeEvent(event)

        def fail(self, message: str):
            self.status.setText(message)
            self.siril.log(message, s.LogColor.RED)
            self.siril.error_messagebox(message)

    return Window(siril)


def run_gui_mode() -> None:
    from PyQt6.QtWidgets import QApplication

    siril = connect()
    app = QApplication.instance() or QApplication(sys.argv)
    window = create_window(siril)
    window.show()
    app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AstroSharp on the current Siril image.")
    parser.add_argument("-model", choices=list(MODELS),
                        help="dual-psf, hybrid, psf, astroclean or star-mask")
    parser.add_argument("-psf", type=float, help="PSF FWHM in pixels (1-8, steps of 0.25); "
                        "for dual-psf, the one used away from stars")
    parser.add_argument("-starpsf", type=float, help="dual-psf: PSF FWHM for the stars")
    parser.add_argument("-aggressiveness", type=float, help="blend amount (0-1)")
    parser.add_argument("-mode", choices=list(MODES), help="seamless or parity")
    parser.add_argument("-chunksize", type=int,
                        help="chunk size for hybrid and dual-psf in parity mode")
    parser.add_argument("-sensitivity", type=float,
                        help="star mask: percentile above which pixels are stars (0-100)")
    parser.add_argument("-fixmask", type=int,
                        help="star mask: leave out the regions numbered up to this")
    parser.add_argument("-feathering", type=float, help="star mask: edge blur sigma (px)")
    parser.add_argument("-blend", type=float, help="star mask: beta curve shape a")
    parser.add_argument("-blend2", type=float, help="star mask: beta curve shape b")
    parser.add_argument("-executable", help="path to astrosharp-cli (or AstroSharp.app)")
    args = parser.parse_args()
    if len(sys.argv) > 1:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
