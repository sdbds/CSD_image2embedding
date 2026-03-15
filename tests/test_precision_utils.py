import unittest

import torch

from precision_utils import resolve_amp_dtype


class ResolveAmpDtypeTests(unittest.TestCase):
    def test_auto_uses_fp16_on_cuda(self):
        self.assertEqual(resolve_amp_dtype("cuda", "auto"), torch.float16)

    def test_auto_disables_amp_on_cpu(self):
        self.assertIsNone(resolve_amp_dtype("cpu", "auto"))

    def test_fp32_disables_amp(self):
        self.assertIsNone(resolve_amp_dtype("cuda", "fp32"))

    def test_bf16_maps_to_bfloat16(self):
        self.assertEqual(resolve_amp_dtype("cuda", "bf16"), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
