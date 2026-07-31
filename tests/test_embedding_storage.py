import unittest

import numpy as np

from csd_image2embedding.data.lance import build_embedding_table


class BuildEmbeddingTableTests(unittest.TestCase):
    def test_stores_raw_embeddings_alongside_2d_projection_columns(self):
        table = build_embedding_table(
            paths=["a.jpg", "b.jpg"],
            previews=["img-a", "img-b"],
            style_embeddings=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            content_embeddings=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
            style_projection=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            content_projection=np.array([[1.1, 1.2], [1.3, 1.4]], dtype=np.float32),
        )

        self.assertEqual(
            table.column_names,
            [
                "path",
                "image",
                "style_embedding",
                "content_embedding",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
        )
        self.assertEqual(table["style_embedding"].to_pylist(), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(
            table["content_embedding"].to_pylist(),
            [[5.0, 6.0], [7.0, 8.0]],
        )


if __name__ == "__main__":
    unittest.main()
