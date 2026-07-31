import numpy as np
import pandas as pd

from csd_image2embedding.clustering.analysis import GenericClusteringResult
from csd_image2embedding.dashboard.figures import create_cluster_figure


def test_cluster_figure_uses_requested_coordinate_set_and_representatives():
    data = pd.DataFrame(
        {
            "x1": [100.0, 200.0],
            "y1": [300.0, 400.0],
            "x2": [1.0, 2.0],
            "y2": [3.0, 4.0],
            "image": ["image-a", "image-b"],
        }
    )
    result = GenericClusteringResult([0, 1], "test")

    figure = create_cluster_figure(
        data,
        result,
        representatives=["representative-a", "representative-b"],
        centers=np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32),
        title="Content",
        feature_set="2",
    )

    assert list(figure.data[0].x) == [1.0, 2.0]
    assert len(figure.layout.images) == 2
    assert figure.layout.title.text == "Content"
