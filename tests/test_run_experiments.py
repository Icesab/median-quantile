import csv
import tempfile
import unittest
from pathlib import Path

from run_experiments import DEFAULT_IMAGES, DEFAULT_METHODS, DEFAULT_PEAKS, DEFAULT_SEEDS, run_experiments


class RunExperimentsTests(unittest.TestCase):
    def test_default_experiment_grid_has_expected_size(self):
        self.assertEqual(len(DEFAULT_IMAGES) * len(DEFAULT_PEAKS) * len(DEFAULT_SEEDS) * len(DEFAULT_METHODS), 20)

    def test_run_experiments_writes_metrics_and_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            rows = run_experiments(
                images=["camera"],
                peaks=[8],
                seeds=[0],
                methods=["noisy", "mean3x3", "median3x3", "method1", "methodA_local", "method2"],
                output_dir=output_dir,
            )

            self.assertEqual(len(rows), 6)

            metrics_path = output_dir / "metrics.csv"
            self.assertTrue(metrics_path.exists())

            with metrics_path.open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

            self.assertEqual(len(csv_rows), 6)

            case_dir = output_dir / "images" / "camera" / "peak_8" / "seed_0"
            expected_files = [
                case_dir / "clean.png",
                case_dir / "noisy.png",
                case_dir / "mean3x3.png",
                case_dir / "median3x3.png",
                case_dir / "method1.png",
                case_dir / "methodA_local.png",
                case_dir / "method2.png",
            ]
            for path in expected_files:
                self.assertTrue(path.exists(), msg=f"missing {path}")

            for row in csv_rows:
                self.assertTrue(Path(row["clean_path"]).exists())
                self.assertTrue(Path(row["noisy_path"]).exists())
                self.assertTrue(Path(row["denoised_path"]).exists())


if __name__ == "__main__":
    unittest.main()
