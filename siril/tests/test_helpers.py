"""Tests for AstroSharp.py's helpers, run outside Siril (sirilpy is stubbed).

The CLI integration test needs the AstroSharp CLI and a sample FITS file:
  ASTROSHARP_CLI   path to astrosharp-cli (default: the one in /Applications/AstroSharp.app)
  ASTROSHARP_FITS  a stretched RGB or grey FITS file (Siril output)
It is skipped when either is missing.
"""

import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# Stub sirilpy so the script can be imported without Siril.
sys.modules["sirilpy"] = types.SimpleNamespace(
    ensure_installed=lambda *a: None,
    LogColor=types.SimpleNamespace(DEFAULT="default", RED="red", SALMON="salmon",
                                   GREEN="green", BLUE="blue"),
)
_spec = importlib.util.spec_from_file_location(
    "AstroSharp", Path(__file__).resolve().parent.parent / "AstroSharp.py")
A = importlib.util.module_from_spec(_spec)
sys.modules["AstroSharp"] = A  # dataclasses look the module up while it loads
_spec.loader.exec_module(A)


class FitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        for shape in [(5, 7), (3, 4, 6)]:
            data = rng.random(shape, dtype=np.float32)
            path = self.dir / "x.fit"
            A.write_fits(str(path), data)
            self.assertEqual(path.stat().st_size % 2880, 0)
            back = A.read_fits(str(path))
            self.assertEqual(back.dtype, np.float32)
            np.testing.assert_array_equal(back, data)

    def test_header_layout(self):
        path = self.dir / "x.fit"
        A.write_fits(str(path), np.zeros((3, 2, 5), np.float32))
        cards = [path.read_bytes()[i:i + 80].decode() for i in range(0, 2880, 80)]
        self.assertEqual(cards[0][:30], "SIMPLE  =                    T")
        self.assertEqual(cards[1][:30], "BITPIX  =                  -32")
        self.assertEqual([c[:30] for c in cards[2:6]], [
            "NAXIS   =                    3",
            "NAXIS1  =                    5",
            "NAXIS2  =                    2",
            "NAXIS3  =                    3",
        ])
        self.assertTrue(cards[6].startswith("ROWORDER= 'BOTTOM-UP'"))
        self.assertTrue(cards[7].startswith("END "))

    def test_top_down_files_are_flipped_to_stored_order(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        path = self.dir / "x.fit"
        A.write_fits(str(path), data)
        raw = bytearray(path.read_bytes())
        raw[400:480] = "ROWORDER= 'TOP-DOWN'".ljust(80).encode()  # 6th card (2-axis image)
        path.write_bytes(bytes(raw))
        np.testing.assert_array_equal(A.read_fits(str(path)), data[::-1])

    def test_rejects_other_formats(self):
        path = self.dir / "x.fit"
        A.write_fits(str(path), np.zeros((2, 2), np.float32))
        raw = bytearray(path.read_bytes())
        raw[80:160] = "BITPIX  =                   16".ljust(80).encode()
        path.write_bytes(bytes(raw))
        with self.assertRaises(ValueError):
            A.read_fits(str(path))
        with self.assertRaises(ValueError):
            A.write_fits(str(path), np.zeros((2, 2, 2), np.float32))


class ConversionTests(unittest.TestCase):
    def test_uint16_round_trip_is_exact(self):
        pixels = np.arange(65536, dtype=np.uint16)
        np.testing.assert_array_equal(A.from_float(A.to_float(pixels), np.uint16), pixels)

    def test_float_passes_through_and_is_clipped(self):
        data = np.array([-0.1, 0.25, 1.5], np.float32)
        self.assertIs(A.to_float(data), data)
        np.testing.assert_array_equal(A.from_float(data, np.float32), [0.0, 0.25, 1.0])


class CommandTests(unittest.TestCase):
    def test_build_command(self):
        cmd = A.build_command("cli", "in.fit", "out.fit", A.Params())
        self.assertEqual(cmd, ["cli", "in.fit", "out.fit", "--model", "hybrid", "--psf", "3",
                               "--aggressiveness", "0.66", "--mode", "seamless",
                               "--depth", "32"])
        parity = A.Params(mode="parity", chunk_size=150, psf=2.25)
        cmd = A.build_command("cli", "i", "o", parity)
        self.assertIn("2.25", cmd)
        self.assertEqual(cmd[-2:], ["--chunk-size", "150"])

    def test_history_text(self):
        self.assertEqual(A.history_text(A.Params()), "AstroSharp: Hybrid PSF 3 aggr 0.66")
        self.assertEqual(A.history_text(A.Params(model="astroclean", aggressiveness=0.5)),
                         "AstroSharp: AstroClean aggr 0.50")
        longest = A.Params(mode="parity", chunk_size=750, psf=7.75)
        self.assertLessEqual(len(A.history_text(longest).encode()), 70)

    def test_params_from_config(self):
        p = A.Params.from_dict({"model": "psf", "psf": "2.5", "aggressiveness": None,
                                "mode": "bogus", "chunk_size": 200})
        self.assertEqual(p, A.Params(model="psf", psf=2.5, chunk_size=200))

    def test_find_executable(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "astrosharp-cli"
            exe.write_text("#!/bin/sh\necho astrosharp-cli 9.9.9\n")
            exe.chmod(0o755)
            self.assertEqual(A.find_executable(str(exe)), str(exe))
            app = Path(d) / "AstroSharp.app" / "Contents" / "MacOS"
            app.mkdir(parents=True)
            (app / "astrosharp-cli").symlink_to(exe)
            self.assertEqual(A.find_executable(str(Path(d) / "AstroSharp.app")),
                             str(app / "astrosharp-cli"))
            self.assertEqual(A.cli_version(str(exe)), (9, 9, 9))
            missing = str(Path(d) / "nope")
            default = A.DEFAULT_CLI if os.access(A.DEFAULT_CLI, os.X_OK) else None
            self.assertEqual(A.find_executable(missing), default)


CLI = os.environ.get("ASTROSHARP_CLI", A.DEFAULT_CLI)
HAVE_CLI = os.access(CLI, os.X_OK)
SAMPLE = os.environ.get("ASTROSHARP_FITS", "")


@unittest.skipUnless(os.access(CLI, os.X_OK) and os.path.isfile(SAMPLE),
                     "needs ASTROSHARP_CLI and ASTROSHARP_FITS")
class CliIntegrationTests(unittest.TestCase):
    """Processing data the way the Siril script does must match running the CLI on the
    original file, i.e. row order and data handling agree with Siril's."""

    def run_cli(self, inp, out, p):
        subprocess.run(A.build_command(CLI, inp, out, p), check=True, capture_output=True)
        return A.read_fits(out)

    def test_matches_direct_cli_run(self):
        # Siril hands the script the image in stored (bottom-up) order, like read_fits.
        siril_pixels = A.read_fits(SAMPLE)
        with tempfile.TemporaryDirectory() as d:
            inp, out, ref = (os.path.join(d, n) for n in ("in.fit", "out.fit", "ref.fit"))
            A.write_fits(inp, siril_pixels)
            p = A.Params()
            via_script = self.run_cli(inp, out, p)
            direct = self.run_cli(SAMPLE, ref, p)
            np.testing.assert_array_equal(via_script, direct)
            self.assertFalse(np.array_equal(via_script, siril_pixels), "image was not processed")

            identity = self.run_cli(inp, out, A.Params(model="psf", aggressiveness=0.0))
            np.testing.assert_array_equal(A.from_float(identity, np.float32), siril_pixels)


if __name__ == "__main__":
    unittest.main()


class FakeSiril:
    """Records what the script does to the 'current image'."""

    def __init__(self, pixels):
        self.pixels = pixels
        self.logs, self.undo, self.result = [], [], None

    def is_image_loaded(self):
        return True

    def get_image_pixeldata(self):
        return self.pixels.copy()

    @contextlib.contextmanager
    def image_lock(self):
        yield

    def undo_save_state(self, message):
        self.undo.append(message)

    def set_image_pixeldata(self, data):
        self.result = data

    def log(self, message, color=None):
        self.logs.append((message, color))

    # Used by the window.
    def get_siril_configdir(self):
        return self.configdir

    def update_progress(self, message, fraction):
        self.progress = (message, fraction)

    def reset_progress(self):
        self.progress = None

    def error_messagebox(self, message):
        self.errors = getattr(self, "errors", []) + [message]


@unittest.skipUnless(HAVE_CLI, "needs ASTROSHARP_CLI")
class JobTests(unittest.TestCase):
    """The Siril-side flow (Job) with a fake Siril and the real CLI."""

    def run_job(self, pixels, params):
        siril = FakeSiril(pixels)
        job = A.Job(siril, CLI, params)
        code, stderr = job.run()
        try:
            job.finish(siril, code, stderr)
        finally:
            self.assertFalse(os.path.exists(job.tmp.name), "temp files left behind")
        return siril

    def test_float_rgb_is_processed_with_undo_point(self):
        pixels = np.random.default_rng(2).uniform(0.2, 0.8, (3, 40, 50)).astype(np.float32)
        siril = self.run_job(pixels, A.Params())
        self.assertEqual((siril.result.shape, siril.result.dtype), (pixels.shape, np.float32))
        self.assertFalse(np.array_equal(siril.result, pixels))
        self.assertEqual(siril.undo, ["AstroSharp: Hybrid PSF 3 aggr 0.66"])
        self.assertTrue(any(c == "green" for _, c in siril.logs))

    def test_uint16_mono_identity(self):
        pixels = np.random.default_rng(3).integers(10000, 60000, (30, 40)).astype(np.uint16)
        siril = self.run_job(pixels, A.Params(model="psf", aggressiveness=0.0))
        self.assertEqual(siril.result.dtype, np.uint16)
        np.testing.assert_array_equal(siril.result, pixels)

    def test_linear_note_is_logged_as_warning(self):
        pixels = np.full((3, 30, 30), 0.01, np.float32)
        siril = self.run_job(pixels, A.Params())
        notes = [m for m, c in siril.logs if c == "salmon"]
        self.assertTrue(notes and "looks linear" in notes[0], siril.logs)

    def test_cli_failure_is_reported_and_image_untouched(self):
        siril = FakeSiril(np.full((20, 20), 0.5, np.float32))
        job = A.Job(siril, CLI, A.Params(psf=3.1))  # not on the 0.25 grid
        code, stderr = job.run()
        self.assertNotEqual(code, 0)
        with self.assertRaisesRegex(RuntimeError, "PSF FWHM"):
            job.finish(siril, code, stderr)
        self.assertIsNone(siril.result)
        self.assertEqual(siril.undo, [])


try:
    import PyQt6  # noqa: F401
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT and HAVE_CLI, "needs PyQt6 and ASTROSHARP_CLI")
class WindowTests(unittest.TestCase):
    """Smoke test of the settings window, off-screen, with a fake Siril."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_apply_runs_in_background_and_updates_image(self):
        with tempfile.TemporaryDirectory() as d:
            pixels = np.random.default_rng(4).uniform(0.2, 0.8, (3, 32, 40)).astype(np.float32)
            siril = FakeSiril(pixels)
            siril.configdir = d
            w = A.create_window(siril)
            w.exe.setText(CLI)
            w.update_version()
            self.assertTrue(w.version.text().startswith("AstroSharp "), w.version.text())

            # Row visibility follows the model and mode.
            w.model.setCurrentIndex(list(A.MODELS).index("astroclean"))
            self.assertFalse(w.psf.isEnabled())
            w.model.setCurrentIndex(list(A.MODELS).index("hybrid"))
            w.mode.setCurrentIndex(list(A.MODES).index("parity"))
            self.assertTrue(w.form.isRowVisible(w.chunk))
            w.mode.setCurrentIndex(list(A.MODES).index("seamless"))
            self.assertFalse(w.form.isRowVisible(w.chunk))
            w.psf.setValue(2.5)
            w.aggr.setValue(0.5)
            self.assertEqual(w.params(), A.Params(psf=2.5, aggressiveness=0.5))

            w.apply()
            self.assertFalse(w.apply_btn.isEnabled())
            w.worker.wait(30000)
            for _ in range(50):  # deliver the queued "done" signal on this thread
                self.app.processEvents()
                if w.apply_btn.isEnabled():
                    break
            self.assertTrue(w.apply_btn.isEnabled())
            self.assertEqual(siril.undo, ["AstroSharp: Hybrid PSF 2.5 aggr 0.50"])
            self.assertEqual(siril.result.shape, pixels.shape)
            self.assertTrue(w.status.text().startswith("Done"), w.status.text())
            with open(os.path.join(d, A.CONFIG_NAME)) as f:
                self.assertIn('"psf": 2.5', f.read())
            w.close()
