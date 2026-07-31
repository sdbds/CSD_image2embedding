import os
import unittest
from unittest.mock import patch

from csd_image2embedding.workflow import configure_runtime_env


class RuntimeEnvTests(unittest.TestCase):
    def test_configure_runtime_env_disables_transformers_tensorflow_backend(self):
        with patch.dict(os.environ, {}, clear=True):
            configure_runtime_env()

            self.assertEqual(os.environ["USE_TF"], "0")
            self.assertEqual(os.environ["TRANSFORMERS_NO_TF"], "1")


if __name__ == "__main__":
    unittest.main()
