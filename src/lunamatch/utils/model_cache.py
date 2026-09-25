"""Writable local cache for optional pretrained model weights."""

import getpass
import os
from pathlib import Path
import tempfile


def set_model_cache() -> Path:
    """Configure torch.hub to use the LunaMatch model cache directory."""
    import torch

    cache = Path(os.environ.get("LUNAMATCH_MODEL_CACHE",
                                str(Path(tempfile.gettempdir()) / f"lunamatch-models-{getpass.getuser()}")))
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    torch.hub.set_dir(str(cache))
    return cache
