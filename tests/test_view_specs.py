import unittest

from view_specs import build_default_view_specs


class ViewSpecsTests(unittest.TestCase):
    def test_default_views_include_finch_style_and_content(self):
        titles, params_list = build_default_view_specs("CSD", 40)

        self.assertEqual(
            titles,
            [
                "[CSD] KMeans_style",
                "[CSD] HDBSCAN_style",
                "[CSD] FINCH_style",
                "[CSD] KMeans_content",
                "[CSD] HDBSCAN_content",
                "[CSD] FINCH_content",
            ],
        )
        self.assertEqual(
            [params["clusterer"] for params in params_list],
            ["kmeans", "hdbscan", "finch", "kmeans", "hdbscan", "finch"],
        )
        self.assertEqual(params_list[2]["k"], 40)
        self.assertEqual(params_list[5]["k"], 40)


if __name__ == "__main__":
    unittest.main()
