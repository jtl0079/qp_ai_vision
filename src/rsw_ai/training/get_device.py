"""
src/rsw_ai/training/get_device.py
------------------------------------
"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[INFO] No GPU found -- running on CPU (slower). "
          "Use a GPU runtime (e.g. Colab/Kaggle) for a big speedup.")
    return torch.device("cpu")
