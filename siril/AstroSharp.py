# AstroSharp for Siril
#
# Runs AstroSharp (https://github.com/deepskydetail/AstroSharp, Rust port) on the image
# currently loaded in Siril: sharpen (PSF), denoise (AstroClean) or both (Hybrid). The
# image is updated in place with an undo point, so Siril's Undo restores it.
#
# AstroSharp is meant as a final touch-up on a stretched (non-linear) image.
#
# Requirements:
#   - Siril 1.4 or later
#   - AstroSharp 0.3.0 or later; its CLI is found in
#     /Applications/AstroSharp.app/Contents/MacOS/astrosharp-cli (or choose it in the window)
#
# Usage:
#   - From Siril's Scripts menu: opens a window with the settings.
#   - From a Siril script (.ssf) or the command line, e.g.:
#       pyscript AstroSharp.py -model hybrid -psf 3 -aggressiveness 0.66
#     Options not given use the settings last saved from the window.
#
# Version: 0.1.0
# 0.1.0: First version.

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Optional

import sirilpy as s

s.ensure_installed("PyQt6")

import numpy as np  # noqa: E402 (installed with sirilpy)

VERSION = "0.1.0"
DEFAULT_CLI = "/Applications/AstroSharp.app/Contents/MacOS/astrosharp-cli"
MIN_CLI_VERSION = (0, 3, 0)
CONFIG_NAME = "astrosharp.conf"
MODELS = {"hybrid": "Hybrid", "psf": "PSF", "astroclean": "AstroClean"}
MODES = {"seamless": "Seamless", "parity": "R parity"}
FITS_BLOCK = 2880


@dataclass
class Params:
    model: str = "hybrid"
    psf: float = 3.0
    aggressiveness: float = 0.66
    mode: str = "seamless"
    chunk_size: int = 325

    @classmethod
    def from_dict(cls, d: dict) -> Params:
        p = cls()
        if d.get("model") in MODELS:
            p.model = d["model"]
        if d.get("mode") in MODES:
            p.mode = d["mode"]
        for key, kind in (("psf", float), ("aggressiveness", float), ("chunk_size", int)):
            try:
                setattr(p, key, kind(d[key]))
            except (KeyError, TypeError, ValueError):
                pass
        return p


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


def fmt(x: float) -> str:
    return f"{x:g}"


def build_command(exe: str, inp: str, out: str, p: Params) -> list[str]:
    cmd = [exe, inp, out, "--model", p.model, "--psf", fmt(p.psf),
           "--aggressiveness", fmt(p.aggressiveness), "--mode", p.mode, "--depth", "32"]
    if p.model == "hybrid" and p.mode == "parity":
        cmd += ["--chunk-size", str(p.chunk_size)]
    return cmd


def history_text(p: Params) -> str:
    """Undo/HISTORY entry (Siril keeps at most 70 bytes)."""
    parts = [f"AstroSharp: {MODELS[p.model]}"]
    if p.model != "astroclean":
        parts.append(f"PSF {fmt(p.psf)}")
    parts.append(f"aggr {p.aggressiveness:.2f}")
    if p.model == "hybrid" and p.mode == "parity":
        parts.append(f"R parity chunk {p.chunk_size}")
    return " ".join(parts)[:70]


def resolve_executable(path: str) -> str:
    """Accepts the CLI itself or the AstroSharp.app bundle containing it."""
    if path.endswith(".app"):
        return os.path.join(path, "Contents", "MacOS", "astrosharp-cli")
    return path


def find_executable(configured: Optional[str]) -> Optional[str]:
    for candidate in (configured, DEFAULT_CLI):
        if candidate:
            candidate = resolve_executable(candidate)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def cli_version(exe: str) -> Optional[tuple[int, int, int]]:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", out.stdout)
    return tuple(int(g) for g in m.groups()) if m else None


def check_executable(configured: Optional[str]) -> str:
    """The usable CLI path, or raises RuntimeError explaining what is wrong."""
    exe = find_executable(configured)
    if exe is None:
        raise RuntimeError(
            f"AstroSharp CLI not found. Install AstroSharp.app in /Applications or choose "
            f"the CLI ({DEFAULT_CLI}) in the AstroSharp window.")
    version = cli_version(exe)
    if version is None or version < MIN_CLI_VERSION:
        need = ".".join(map(str, MIN_CLI_VERSION))
        raise RuntimeError(f"{exe} is not AstroSharp {need} or later.")
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
    """One run: the current image written to a temporary FITS file, and back."""

    def __init__(self, siril, exe: str, params: Params):
        if not siril.is_image_loaded():
            raise RuntimeError("No image is loaded in Siril.")
        pixels = siril.get_image_pixeldata()
        self.shape, self.dtype = pixels.shape, pixels.dtype
        data = to_float(pixels)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        self.params = params
        self.tmp = tempfile.TemporaryDirectory(prefix="astrosharp-")
        self.inp = os.path.join(self.tmp.name, "in.fit")
        self.out = os.path.join(self.tmp.name, "out.fit")
        write_fits(self.inp, data)
        self.command = build_command(exe, self.inp, self.out, params)
        self.started = time.monotonic()

    def run(self) -> tuple[int, str]:
        """Runs the CLI (safe to call from a worker thread). Returns (exit code, stderr)."""
        try:
            proc = subprocess.run(self.command, capture_output=True, text=True)
            return proc.returncode, proc.stderr
        except OSError as e:
            return -1, str(e)

    def finish(self, siril, code: int, stderr: str) -> None:
        """Logs the CLI output and, on success, replaces the image with an undo point."""
        try:
            for line in stderr.splitlines():
                if line.startswith("note:"):
                    siril.log(f"AstroSharp {line}", s.LogColor.SALMON)
            if code != 0:
                raise RuntimeError(f"AstroSharp failed: {stderr.strip() or f'exit code {code}'}")
            result = from_float(read_fits(self.out).reshape(self.shape), self.dtype)
            with siril.image_lock():
                siril.undo_save_state(history_text(self.params))
                siril.set_image_pixeldata(result)
            siril.log(f"{history_text(self.params)} applied in "
                      f"{time.monotonic() - self.started:.1f} s", s.LogColor.GREEN)
        finally:
            self.tmp.cleanup()


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
                       ("chunk_size", args.chunksize)):
        if value is not None:
            setattr(p, key, value)
    try:
        exe = check_executable(args.executable or config.get("executable"))
        if args.executable:
            save_config(siril, exe, Params.from_dict(config))
        job = Job(siril, exe, p)
        siril.update_progress("AstroSharp: processing…", -1.0)
        try:
            code, stderr = job.run()
        finally:
            siril.reset_progress()
        job.finish(siril, code, stderr)
    except Exception as e:  # report every failure in Siril's log
        siril.log(str(e), s.LogColor.RED)
        sys.exit(1)


def create_window(siril):
    """Builds the settings window (PyQt6 is imported here so the helpers work without it)."""
    from PyQt6.QtCore import QThread, pyqtSignal
    from PyQt6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QFileDialog,
                                 QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                                 QSpinBox, QVBoxLayout, QWidget)

    class Worker(QThread):
        done = pyqtSignal(int, str)

        def __init__(self, job: Job):
            super().__init__()
            self.job = job

        def run(self):
            code, stderr = self.job.run()
            self.done.emit(code, stderr)

    class Window(QWidget):
        def __init__(self, siril):
            super().__init__()
            self.siril = siril
            self.worker = None
            self.job = None
            config = load_config(siril)
            p = Params.from_dict(config)
            self.setWindowTitle(f"AstroSharp – v{VERSION}")

            self.model = QComboBox()
            for key, name in MODELS.items():
                self.model.addItem(name, key)
            self.model.setCurrentIndex(list(MODELS).index(p.model))
            self.psf = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=8.0, singleStep=0.25)
            self.psf.setValue(p.psf)
            self.psf.setToolTip("70% of the PSF in pixels: measure the FWHM of a small, "
                                "unsaturated star and divide by 2.35.")
            self.aggr = QDoubleSpinBox(decimals=2, minimum=0.01, maximum=1.0, singleStep=0.01)
            self.aggr.setValue(p.aggressiveness)
            self.aggr.setToolTip("Blend between the model output (1) and the input (0).")
            self.mode = QComboBox()
            for key, name in MODES.items():
                self.mode.addItem(name, key)
            self.mode.setCurrentIndex(list(MODES).index(p.mode))
            self.mode.setToolTip("Seamless: whole-image processing at full precision. "
                                 "R parity: reproduces the original R app exactly.")
            self.chunk = QSpinBox(minimum=50, maximum=750)
            self.chunk.setValue(p.chunk_size)

            self.exe = QLineEdit(config.get("executable") or find_executable(None) or "")
            browse = QPushButton("Browse…")
            browse.clicked.connect(self.browse)
            self.version = QLabel()
            exe_row = QHBoxLayout()
            exe_row.addWidget(self.exe)
            exe_row.addWidget(browse)

            self.form = QFormLayout()
            self.form.addRow("Model", self.model)
            self.form.addRow("PSF FWHM (px)", self.psf)
            self.form.addRow("Aggressiveness", self.aggr)
            self.form.addRow("Mode", self.mode)
            self.form.addRow("Chunk size", self.chunk)
            self.form.addRow("AstroSharp CLI", exe_row)
            self.form.addRow("", self.version)

            hint = QLabel("Each Apply processes the current image; use Siril's Undo to try "
                          "other settings. AstroSharp is meant for stretched images.")
            hint.setWordWrap(True)
            self.status = QLabel()
            self.status.setWordWrap(True)
            self.apply_btn = QPushButton("Apply")
            self.apply_btn.setDefault(True)
            self.apply_btn.clicked.connect(self.apply)
            close = QPushButton("Close")
            close.clicked.connect(self.close)
            buttons = QHBoxLayout()
            buttons.addStretch()
            buttons.addWidget(self.apply_btn)
            buttons.addWidget(close)

            layout = QVBoxLayout(self)
            layout.addLayout(self.form)
            layout.addWidget(hint)
            layout.addWidget(self.status)
            layout.addLayout(buttons)

            self.model.currentIndexChanged.connect(self.update_rows)
            self.mode.currentIndexChanged.connect(self.update_rows)
            self.exe.editingFinished.connect(self.update_version)
            self.update_rows()
            self.update_version()

        def params(self) -> Params:
            return Params(self.model.currentData(), self.psf.value(), self.aggr.value(),
                          self.mode.currentData(), self.chunk.value())

        def update_rows(self):
            hybrid = self.model.currentData() == "hybrid"
            self.psf.setEnabled(self.model.currentData() != "astroclean")
            self.form.setRowVisible(self.mode, hybrid)
            self.form.setRowVisible(self.chunk, hybrid and self.mode.currentData() == "parity")

        def update_version(self):
            exe = find_executable(self.exe.text())
            version = cli_version(exe) if exe else None
            if version is None:
                self.version.setText("CLI not found")
            else:
                ok = "" if version >= MIN_CLI_VERSION else " (too old; 0.3.0 or later needed)"
                self.version.setText(f"AstroSharp {'.'.join(map(str, version))}{ok}")

        def browse(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose the AstroSharp CLI (or AstroSharp.app)",
                os.path.dirname(self.exe.text()) or "/Applications")
            if path:
                self.exe.setText(resolve_executable(path))
                self.update_version()

        def apply(self):
            p = self.params()
            try:
                exe = check_executable(self.exe.text())
                save_config(self.siril, exe, p)
                job = Job(self.siril, exe, p)
            except Exception as e:
                self.fail(str(e))
                return
            self.apply_btn.setEnabled(False)
            self.status.setText("Processing…")
            self.siril.update_progress("AstroSharp: processing…", -1.0)
            self.job = job
            self.worker = Worker(job)
            # A bound method of this (main-thread) widget, so Qt delivers it on the GUI thread.
            self.worker.done.connect(self.finished)
            self.worker.start()

        def finished(self, code: int, stderr: str):
            job, self.job = self.job, None
            self.siril.reset_progress()
            self.apply_btn.setEnabled(True)
            try:
                job.finish(self.siril, code, stderr)
                self.status.setText(f"Done: {history_text(job.params)}")
            except s.ProcessingThreadBusyError:
                self.fail("Siril is busy with another task; try again when it has finished.")
            except s.ImageDialogOpenError:
                self.fail("Close Siril's open processing dialog and try again.")
            except Exception as e:
                self.fail(str(e))

        def closeEvent(self, event):
            # Never destroy a running worker thread.
            if self.worker is not None and self.worker.isRunning():
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
    parser.add_argument("-model", choices=list(MODELS), help="hybrid, psf or astroclean")
    parser.add_argument("-psf", type=float, help="PSF FWHM in pixels (1-8, steps of 0.25)")
    parser.add_argument("-aggressiveness", type=float, help="blend amount (0-1)")
    parser.add_argument("-mode", choices=list(MODES), help="seamless or parity")
    parser.add_argument("-chunksize", type=int, help="chunk size for hybrid in parity mode")
    parser.add_argument("-executable", help="path to astrosharp-cli (or AstroSharp.app)")
    args = parser.parse_args()
    if len(sys.argv) > 1:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()
