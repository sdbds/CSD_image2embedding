import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from process_image import classify_images


class DummyResult:
    def __init__(self, labels, algorithm_name, noise_label=None):
        self.labels_ = labels
        self.algorithm_name = algorithm_name
        self.noise_label = noise_label


class ProcessImageTests(unittest.TestCase):
    def test_classify_images_supports_sparse_generic_cluster_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src1 = root / "a.jpg"
            src2 = root / "b.jpg"
            src3 = root / "c.jpg"
            for src in (src1, src2, src3):
                src.write_bytes(b"test")

            data = pd.DataFrame({"path": [str(src1), str(src2), str(src3)]})
            result = DummyResult(labels=[5, 5, 7], algorithm_name="finch")

            classify_images(
                data,
                result,
                SimpleNamespace(symlink=False),
                str(root / "out"),
            )

            self.assertTrue((root / "out" / "class_5").exists())
            self.assertTrue((root / "out" / "class_7").exists())
            self.assertFalse((root / "out" / "noise").exists())


if __name__ == "__main__":
    unittest.main()
