"""Trainable lunar patch representations.

The training utilities are optional and require the ``learned`` dependency
extra. Checkpoints are research artifacts and are never treated as measured
registration accuracy.
"""

from .descriptor import LunarPatchDescriptor

__all__ = ["LunarPatchDescriptor"]
