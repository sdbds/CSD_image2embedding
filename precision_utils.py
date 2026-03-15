from contextlib import nullcontext

import torch


def get_device_type(device) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def resolve_amp_dtype(device, precision: str = "auto"):
    device_type = get_device_type(device)
    precision = precision.lower()

    if precision == "fp32":
        return None
    if precision == "auto":
        if device_type == "cuda":
            return torch.float16
        return None
    if precision == "fp16":
        if device_type != "cuda":
            return None
        return torch.float16
    if precision == "bf16":
        if device_type in {"cuda", "cpu"}:
            return torch.bfloat16
        return None

    raise ValueError(f"Unsupported precision mode: {precision}")


def autocast_context(device, amp_dtype):
    device_type = get_device_type(device)
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device_type, dtype=amp_dtype)
