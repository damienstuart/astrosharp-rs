"""Tests for AstroSharp.py's helpers, run outside Siril (sirilpy is stubbed).

The CLI integration test needs the AstroSharp CLI and a sample FITS file:
  ASTROSHARP_CLI   path to astrosharp-cli (default: the one in /Applications/AstroSharp.app)
  ASTROSHARP_FITS  a stretched RGB or grey FITS file (Siril output)
It is skipped when either is missing.
"""

import contextlib
import importlib.util
import json
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


MASK_OPTIONS = ["--sensitivity", "98", "--fix-mask", "2", "--feathering", "5", "--blend", "9",
                "--blend2", "100"]


class CommandTests(unittest.TestCase):
    def test_build_command(self):
        cmd = A.build_command("cli", "in.fit", "out.fit", A.Params(model="hybrid"))
        self.assertEqual(cmd, ["cli", "in.fit", "out.fit", "--model", "hybrid", "--psf", "3",
                               "--aggressiveness", "0.66", "--mode", "seamless",
                               "--depth", "32"])
        parity = A.Params(model="hybrid", mode="parity", chunk_size=150, psf=2.25)
        cmd = A.build_command("cli", "i", "o", parity)
        self.assertIn("2.25", cmd)
        self.assertEqual(cmd[-2:], ["--chunk-size", "150"])

    def test_build_command_for_the_star_mask_models(self):
        # Dual PSF is the default model.
        cmd = A.build_command("cli", "i", "o", A.Params(star_psf=2.0))
        self.assertEqual(cmd[3:], ["--model", "dual-psf", "--psf", "3", "--aggressiveness",
                                   "0.66", "--mode", "seamless", "--depth", "32",
                                   "--star-psf", "2"] + MASK_OPTIONS)
        cmd = A.build_command("cli", "i", "o", A.Params(mode="parity", chunk_size=200))
        self.assertEqual(cmd[-2:], ["--chunk-size", "200"])
        mask = A.Params(model="star-mask", mode="parity", sensitivity=95.5, fix_mask=0,
                        feathering=2.5, blend=2.0, blend2=20.0)
        cmd = A.build_command("cli", "i", "o", mask)
        self.assertNotIn("--star-psf", cmd)
        self.assertNotIn("--chunk-size", cmd)
        self.assertEqual(cmd[-10:], ["--sensitivity", "95.5", "--fix-mask", "0",
                                     "--feathering", "2.5", "--blend", "2", "--blend2", "20"])

    def test_history_text(self):
        hybrid = A.Params(model="hybrid")
        self.assertEqual(A.history_text(hybrid), "AstroSharp: Hybrid PSF 3 aggr 0.66")
        self.assertEqual(A.settings_text(hybrid), "Hybrid PSF 3 aggr 0.66")
        self.assertEqual(A.history_text(A.Params(model="astroclean", aggressiveness=0.5)),
                         "AstroSharp: AstroClean aggr 0.50")
        longest = A.Params(model="hybrid", mode="parity", chunk_size=750, psf=7.75)
        self.assertLessEqual(len(A.history_text(longest).encode()), 70)

    def test_history_text_for_the_star_mask_models(self):
        self.assertEqual(A.settings_text(A.Params(star_psf=2.0)), "Dual PSF 3/2 aggr 0.66")
        self.assertEqual(A.settings_text(A.Params(model="star-mask")), "Star Mask 98/2/5/9/100")
        self.assertEqual(A.settings_text(A.Params(model="star-mask", mode="parity")),
                         "Star Mask 98/2/5/9/100 parity")
        # Dual PSF lists the mask settings only when they aren't the defaults.
        custom = A.Params(star_psf=2.0, mode="parity", sensitivity=95.0, fix_mask=0,
                          feathering=2.5, blend=2.0, blend2=20.0)
        self.assertEqual(A.history_text(custom),
                         "AstroSharp: Dual PSF 3/2 aggr 0.66 parity chunk 325 mask 95/0/2.5/2/20")
        self.assertLessEqual(len(A.history_text(custom).encode()), 70)

    def test_params_from_config(self):
        p = A.Params.from_dict({"model": "psf", "psf": "2.5", "aggressiveness": None,
                                "mode": "bogus", "chunk_size": 200})
        self.assertEqual(p, A.Params(model="psf", psf=2.5, chunk_size=200))
        p = A.Params.from_dict({"model": "star-mask", "star_psf": 2.25, "sensitivity": "90",
                                "fix_mask": 5, "feathering": "x", "blend2": 50})
        self.assertEqual(p, A.Params(model="star-mask", star_psf=2.25, sensitivity=90.0,
                                     fix_mask=5, blend2=50.0))
        # Settings saved by version 0.2.0 keep their model; the new ones get defaults.
        old = A.Params.from_dict({"model": "hybrid", "psf": 3.0, "aggressiveness": 0.66,
                                  "mode": "seamless", "chunk_size": 325})
        self.assertEqual(old, A.Params(model="hybrid"))

    def test_star_mask_models_need_a_newer_cli(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "astrosharp-cli"
            exe.write_text("#!/bin/sh\necho astrosharp-cli 0.4.6\n")
            exe.chmod(0o755)
            self.assertEqual(A.check_executable(str(exe), "hybrid"), str(exe))
            for model in ("dual-psf", "star-mask"):
                with self.assertRaisesRegex(RuntimeError, "needs AstroSharp 0.5.0"):
                    A.check_executable(str(exe), model)
            exe.write_text("#!/bin/sh\necho astrosharp-cli 0.5.0\n")
            self.assertEqual(A.check_executable(str(exe), "dual-psf"), str(exe))

    def test_grey_star_mask_of_colour_image_fills_every_channel(self):
        mask = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = A.fit_shape(mask, (3, 2, 3))
        self.assertEqual(out.shape, (3, 2, 3))
        for c in range(3):
            np.testing.assert_array_equal(out[c], mask)
        np.testing.assert_array_equal(A.fit_shape(mask, (2, 3)), mask)

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
            # Colour goes through Luv and back, so channels that are exactly 0 come back as
            # about 1e-16, which 32-bit floats can hold; everything else is unchanged.
            np.testing.assert_allclose(A.from_float(identity, np.float32), siril_pixels,
                                       rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()


class FakeSiril:
    """Records what the script does to the 'current image'."""

    def __init__(self, pixels):
        self.pixels = pixels
        self.logs, self.undo, self.result = [], [], None
        self.errors, self.warnings = [], []
        self.configdir = tempfile.mkdtemp()
        self.progress = None

    def is_image_loaded(self):
        return self.pixels is not None

    def get_image_pixeldata(self):
        return self.pixels.copy()

    @contextlib.contextmanager
    def image_lock(self):
        yield

    def undo_save_state(self, message):
        self.undo.append(message)

    def set_image_pixeldata(self, data):
        self.result = data
        self.pixels = data.copy()  # like Siril: this is now the current image

    def log(self, message, color=None):
        self.logs.append((message, color))

    def get_siril_configdir(self):
        return self.configdir

    def update_progress(self, message, fraction):
        self.progress = (message, fraction)

    def reset_progress(self):
        self.progress = None

    def error_messagebox(self, message):
        self.errors.append(message)

    def warning_messagebox(self, message):
        self.warnings.append(message)


class DisplayHelperTests(unittest.TestCase):
    def test_display_u8_flips_rows_and_scales(self):
        pixels = np.array([[0, 65535], [32768, 0]], np.uint16)  # row 0 = bottom
        np.testing.assert_array_equal(A.display_u8(pixels), [[128, 0], [0, 255]])
        rgb = np.zeros((3, 2, 2), np.float32)
        rgb[0, 0, 0] = 1.0  # red, bottom-left
        out = A.display_u8(rgb)
        self.assertEqual(out.shape, (2, 2, 3))
        self.assertEqual(out[1, 0].tolist(), [255, 0, 0])
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_fingerprint(self):
        a = np.random.default_rng(5).random((3, 50, 60), dtype=np.float32)
        self.assertEqual(A.fingerprint(a), A.fingerprint(a.copy()))
        self.assertNotEqual(A.fingerprint(a), A.fingerprint(a * 0.9))
        self.assertNotEqual(A.fingerprint(a), A.fingerprint(a.astype(np.float64)))

    def test_looks_linear(self):
        linear = np.full((3, 40, 40), 0.01, np.float32)
        linear[:, ::10, ::10] = 0.9  # 1% bright "stars"
        self.assertTrue(A.looks_linear(linear))
        stretched = np.random.default_rng(6).uniform(0.05, 0.7, (40, 40)).astype(np.float32)
        self.assertFalse(A.looks_linear(stretched))


@unittest.skipUnless(HAVE_CLI, "needs ASTROSHARP_CLI")
class JobTests(unittest.TestCase):
    """The processing job with a fake Siril and the real CLI."""

    def run_job(self, pixels, params):
        siril = FakeSiril(pixels)
        job = A.Job.from_siril(siril, CLI, params)
        job.run()
        job.log_notes(siril)
        return siril, job

    def test_float_rgb_is_processed_with_undo_point(self):
        pixels = np.random.default_rng(2).uniform(0.2, 0.8, (3, 40, 50)).astype(np.float32)
        siril, job = self.run_job(pixels, A.Params(model="hybrid"))
        self.assertIsNone(siril.result, "running must not touch Siril's image")
        job.apply(siril)
        self.assertEqual((siril.result.shape, siril.result.dtype), (pixels.shape, np.float32))
        self.assertFalse(np.array_equal(siril.result, pixels))
        self.assertEqual(siril.undo, ["AstroSharp: Hybrid PSF 3 aggr 0.66"])
        self.assertTrue(any(c == "green" for _, c in siril.logs))

    def test_dual_psf(self):
        pixels = np.random.default_rng(7).uniform(0.2, 0.8, (3, 40, 50)).astype(np.float32)
        siril, job = self.run_job(pixels, A.Params(star_psf=2.0))
        job.apply(siril)
        self.assertEqual(siril.result.shape, pixels.shape)
        self.assertFalse(np.array_equal(siril.result, pixels))
        self.assertEqual(siril.undo, ["AstroSharp: Dual PSF 3/2 aggr 0.66"])

    def test_star_mask_of_colour_image_stays_rgb(self):
        pixels = np.random.default_rng(8).uniform(0.2, 0.8, (3, 40, 50)).astype(np.float32)
        for mode in ("seamless", "parity"):
            siril, job = self.run_job(pixels, A.Params(model="star-mask", mode=mode))
            self.assertEqual(job.result.shape, pixels.shape, mode)
            np.testing.assert_array_equal(job.result[0], job.result[1])
            np.testing.assert_array_equal(job.result[1], job.result[2])

    def test_uint16_mono_identity(self):
        pixels = np.random.default_rng(3).integers(10000, 60000, (30, 40)).astype(np.uint16)
        siril, job = self.run_job(pixels, A.Params(model="psf", aggressiveness=0.0))
        self.assertEqual(job.result.dtype, np.uint16)
        np.testing.assert_array_equal(job.result, pixels)

    def test_linear_note_is_logged_as_warning(self):
        siril, _ = self.run_job(np.full((3, 30, 30), 0.01, np.float32), A.Params())
        notes = [m for m, c in siril.logs if c == "salmon"]
        self.assertTrue(notes and "looks linear" in notes[0], siril.logs)

    def test_cli_failure_is_reported_and_image_untouched(self):
        siril, job = self.run_job(np.full((20, 20), 0.5, np.float32), A.Params(psf=3.1))
        self.assertNotEqual(job.code, 0)
        with self.assertRaisesRegex(RuntimeError, "PSF FWHM"):
            job.apply(siril)
        self.assertIsNone(siril.result)
        self.assertEqual(siril.undo, [])


try:
    import PyQt6  # noqa: F401
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT and HAVE_CLI, "needs PyQt6 and ASTROSHARP_CLI")
class WindowTests(unittest.TestCase):
    """The preview window, off-screen, with a fake Siril and the real CLI."""

    @classmethod
    def setUpClass(cls):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.pixels = np.random.default_rng(4).uniform(0.2, 0.8, (3, 32, 40)).astype(np.float32)
        self.siril = FakeSiril(self.pixels)
        self.w = A.create_window(self.siril)
        self.w.exe.setText(CLI)
        self.w.update_version()

    def tearDown(self):
        self.w.close()

    def wait(self):
        """Runs the event loop until no job is running or queued."""
        for _ in range(400):
            self.app.processEvents()
            if not self.w.running() and not self.w.send_when_done and not self.w.rerun:
                self.app.processEvents()
                return
            if self.w.worker is not None:
                self.w.worker.wait(50)
        self.fail("timed out")

    def test_opens_with_original_and_no_processing(self):
        self.assertTrue(self.w.version.text().startswith("AstroSharp "), self.w.version.text())
        self.assertIsNotNone(self.w.view.before)
        self.assertIsNone(self.w.view.after)
        self.assertIsNone(self.w.worker)
        self.assertTrue(self.w.send_btn.isEnabled())

    def select(self, model):
        self.w.model.setCurrentIndex(list(A.MODELS).index(model))

    def test_rows_follow_model_and_mode(self):
        w = self.w
        self.select("astroclean")
        self.assertFalse(w.psf.isEnabled())
        self.assertTrue(w.mask_box.isHidden())
        self.select("hybrid")
        w.mode.setCurrentIndex(list(A.MODES).index("parity"))
        self.assertTrue(w.form.isRowVisible(w.chunk))
        self.assertFalse(w.form.isRowVisible(w.star_psf))
        w.mode.setCurrentIndex(list(A.MODES).index("seamless"))
        self.assertFalse(w.form.isRowVisible(w.chunk))

    def test_rows_for_the_star_mask_models(self):
        w = self.w
        self.assertEqual(w.model.currentData(), "dual-psf", "Dual PSF is the default")
        self.assertTrue(w.form.isRowVisible(w.star_psf))
        self.assertEqual(w.form.labelForField(w.psf).text(), "DSO PSF FWHM (px)")
        self.assertFalse(w.mask_box.isHidden())
        w.mode.setCurrentIndex(list(A.MODES).index("parity"))
        self.assertTrue(w.form.isRowVisible(w.chunk))
        self.select("star-mask")
        self.assertFalse(w.psf.isEnabled() or w.aggr.isEnabled())
        self.assertFalse(w.form.isRowVisible(w.star_psf))
        self.assertTrue(w.form.isRowVisible(w.mode))
        self.assertFalse(w.form.isRowVisible(w.chunk))
        self.assertFalse(w.mask_box.isHidden())
        self.select("psf")
        self.assertEqual(w.form.labelForField(w.psf).text(), "PSF FWHM (px)")
        self.assertFalse(w.form.isRowVisible(w.mode))

    def test_mask_settings_are_used_saved_and_reset(self):
        w = self.w
        w.star_psf.setValue(2.0)
        w.sensitivity.setValue(95.0)
        w.fix_mask.setValue(0)
        w.feathering.setValue(2.5)
        w.blend.setValue(2.0)
        w.blend2.setValue(20.0)
        expected = A.Params(star_psf=2.0, sensitivity=95.0, fix_mask=0, feathering=2.5,
                            blend=2.0, blend2=20.0)
        self.assertEqual(w.params(), expected)
        w.start_preview()
        self.wait()
        self.assertEqual(w.status.text().split(" (")[0],
                         "Preview: Dual PSF 3/2 aggr 0.66 mask 95/0/2.5/2/20")
        with open(os.path.join(self.siril.configdir, A.CONFIG_NAME)) as f:
            self.assertEqual(A.Params.from_dict(json.load(f)), expected)
        w.set_mask(A.Params())
        self.assertEqual(w.params(), A.Params(star_psf=2.0))

    def test_star_mask_preview(self):
        w = self.w
        self.select("star-mask")
        w.start_preview()
        self.wait()
        self.assertIsNotNone(w.view.after)
        self.assertEqual(w.preview.result.shape, self.pixels.shape)
        self.assertEqual(w.status.text().split(" (")[0], "Preview: Star Mask 98/2/5/9/100")

    def test_preview_then_send(self):
        w = self.w
        w.psf.setValue(2.5)
        w.aggr.setValue(0.5)
        w.start_preview()
        self.assertFalse(w.preview_btn.isEnabled())
        self.wait()
        self.assertIsNotNone(w.view.after)
        self.assertEqual(w.status.text().split(" (")[0], "Preview: Dual PSF 2.5/3 aggr 0.50")
        self.assertIsNone(self.siril.result, "preview must not change Siril's image")

        previewed = w.preview.result
        w.send()
        self.assertEqual(self.siril.undo, ["AstroSharp: Dual PSF 2.5/3 aggr 0.50"])
        np.testing.assert_array_equal(self.siril.result, previewed)
        # The sent image becomes the new original.
        self.assertIsNone(w.view.after)
        self.assertEqual(w.before_fp, A.fingerprint(previewed))
        with open(os.path.join(self.siril.configdir, A.CONFIG_NAME)) as f:
            self.assertIn('"psf": 2.5', f.read())

    def test_send_with_changed_settings_processes_first(self):
        w = self.w
        w.start_preview()
        self.wait()
        w.aggr.setValue(0.9)
        self.assertIn("Settings changed", w.status.text())
        w.send()
        self.wait()
        self.assertEqual(self.siril.undo, ["AstroSharp: Dual PSF 3/3 aggr 0.90"])

    def test_refuses_to_send_if_siril_image_changed(self):
        w = self.w
        w.start_preview()
        self.wait()
        self.siril.pixels = self.pixels * 0.5  # changed in Siril after the preview
        w.send()
        self.assertIsNone(self.siril.result)
        self.assertEqual(len(self.siril.warnings), 1)
        self.assertIsNone(w.preview)

    def test_reset_discards_preview_and_reloads_from_siril(self):
        w = self.w
        w.start_preview()
        self.wait()
        self.assertIsNotNone(w.view.after)
        self.siril.pixels = self.pixels * 0.5  # e.g. the user changed the image in Siril
        w.reset()
        self.assertIsNone(w.preview)
        self.assertIsNone(w.view.after)
        self.assertEqual(w.before_fp, A.fingerprint(self.siril.pixels))
        self.assertIsNone(self.siril.result, "reset must not change Siril's image")
        self.assertTrue(w.status.text().startswith("Showing Siril's current image"))

    def test_sliders(self):
        w = self.w
        w.resize(1000, 700)
        w.show()
        self.app.processEvents()
        # Steps of 0.25 (PSF) and 0.01 (aggressiveness), shown in a label that fits.
        w.psf.setValue(2.6)
        self.assertEqual((w.psf.value(), w.psf.label.text()), (2.5, "2.50"))
        w.aggr.slider.setValue(w.aggr.slider.value() + 1)
        self.assertEqual(w.aggr.value(), 0.67)
        w.psf.setValue(8.0)
        w.aggr.setValue(1.0)
        for s in (w.psf, w.aggr):
            self.assertGreaterEqual(s.label.width(), s.label.sizeHint().width(), s.label.text())
            self.assertGreater(s.slider.width(), 100)
        self.assertEqual((w.psf.slider.minimum(), w.psf.slider.maximum()), (4, 32))
        self.assertEqual((w.aggr.slider.minimum(), w.aggr.slider.maximum()), (1, 100))

    def test_live_preview_updates_after_setting_change(self):
        from PyQt6.QtTest import QTest
        w = self.w
        w.live.setChecked(True)
        w.aggr.setValue(0.4)
        for _ in range(100):  # let the debounce timer fire
            if w.worker is not None:
                break
            QTest.qWait(20)
        self.wait()
        self.assertTrue(w.preview_current())
        self.assertIsNone(self.siril.result, "live preview must not change Siril's image")

    def test_divider_drag(self):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest
        w = self.w
        w.resize(1000, 700)
        w.show()
        w.start_preview()
        self.wait()
        w.compare.setChecked(True)
        v = w.view
        y = v.height() // 2
        start = QPoint(int(v.split_x()), y)
        QTest.mousePress(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(v, QPoint(v.width() // 4, y))
        QTest.mouseRelease(v, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                           QPoint(v.width() // 4, y))
        self.assertAlmostEqual(v.split, 0.25, delta=0.02)
