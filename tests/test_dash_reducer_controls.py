import unittest

from dash import dcc, html

from dash_page import build_multi_view_layout


def _find_component_by_id(node, target_id):
    if getattr(node, "id", None) == target_id:
        return node

    children = getattr(node, "children", None)
    if children is None:
        return None
    if not isinstance(children, (list, tuple)):
        children = [children]

    for child in children:
        found = _find_component_by_id(child, target_id)
        if found is not None:
            return found
    return None


class DashReducerControlTests(unittest.TestCase):
    def test_layout_contains_global_reducer_dropdown_with_pacmap_default(self):
        layout = build_multi_view_layout(
            num_views=4,
            reducer_options=[
                {"label": "PaCMAP", "value": "pacmap"},
                {"label": "UMAP", "value": "umap"},
            ],
            default_reducer="pacmap",
            finch_partition_options=[
                {"label": "0", "value": 0},
                {"label": "1", "value": 1},
                {"label": "2", "value": 2},
            ],
            default_finch_partition_index=1,
        )

        dropdown = _find_component_by_id(layout, "reducer-selector")
        finch_dropdown = _find_component_by_id(layout, "finch-partition-selector")
        loading = _find_component_by_id(layout, "reducer-loading")

        self.assertIsInstance(layout, html.Div)
        self.assertIsInstance(dropdown, dcc.Dropdown)
        self.assertEqual(dropdown.value, "pacmap")
        self.assertEqual(dropdown.options[0]["value"], "pacmap")
        self.assertIsInstance(finch_dropdown, dcc.Dropdown)
        self.assertEqual(finch_dropdown.value, 1)
        self.assertEqual(finch_dropdown.options[0]["value"], 0)
        self.assertIsInstance(loading, dcc.Loading)


if __name__ == "__main__":
    unittest.main()
