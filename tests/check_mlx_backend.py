"""MLX backend checks on Apple silicon: pure import, text-only guard, and parity with the HF backend.

Parity runs both backends in float32 at temperature 1 on bundled Typed Decision rows and requires
the same decision for every field and |Δp| <= 0.02. It needs MODEL_CHECKPOINT (a local
Intern-Decision directory) and is skipped when MLX or the checkpoint is unavailable.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path

ROWS = Path(__file__).resolve().parents[1] / "benchmarks/accuracy-v1/typed_decisions/test.jsonl"
CHECKPOINT = os.environ.get("MODEL_CHECKPOINT")
HAS_MLX = importlib.util.find_spec("mlx") is not None and importlib.util.find_spec("mlx_lm") is not None
PARITY_ROWS = int(os.environ.get("MLX_PARITY_ROWS", "8"))


class NoXTuner(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xtuner" or fullname.startswith("xtuner."):
            raise AssertionError("MLX backend imported XTuner")
        return None


sys.meta_path.insert(0, NoXTuner())


def request(line):
    row = json.loads(line)
    return {"state": row["state"], "questions": row["questions"]}


@unittest.skipUnless(HAS_MLX, "mlx and mlx-lm are not installed")
class MLXImportTests(unittest.TestCase):
    def test_pure_mlx_import(self):
        from src.inference.mlx_backend import MLXBackend

        self.assertIsNotNone(MLXBackend)
        self.assertFalse(any(k.startswith("xtuner") for k in sys.modules))


@unittest.skipUnless(HAS_MLX and CHECKPOINT and Path(CHECKPOINT or "").is_dir(), "needs mlx and MODEL_CHECKPOINT")
class MLXParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.inference.engine import DecisionEngine

        cls.mlx = DecisionEngine(CHECKPOINT, backend="mlx", dtype="float32")
        cls.hf = DecisionEngine(CHECKPOINT, backend="hf", device="cpu", dtype="float32")

    def test_images_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "text-only"):
            self.mlx.predict({"state": "x", "images": ["a.png"], "questions": {"q": {"type": "noul"}}})

    def test_parity_with_hf(self):
        lines = ROWS.read_text(encoding="utf-8").splitlines()[:PARITY_ROWS]
        worst = 0.0
        for line in lines:
            req = request(line)
            a = self.hf.predict(req)["answers"]
            b = self.mlx.predict(req)["answers"]
            for field, answer in a.items():
                pa, pb = answer["probabilities"], b[field]["probabilities"]
                self.assertEqual(max(pa, key=pa.get), max(pb, key=pb.get), field)
                worst = max(worst, max(abs(pa[o] - pb[o]) for o in pa))
        self.assertLessEqual(worst, 0.02)
        print(f"\nMLX vs HF float32 on {len(lines)} rows: max |Δp| = {worst:.5f}")


if __name__ == "__main__":
    unittest.main()
