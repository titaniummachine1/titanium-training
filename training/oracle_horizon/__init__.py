"""Guarded oracle-horizon curriculum primitives.

This package describes labels and bounded pilot policy; it does not start
engines, coordinators, producers, or training jobs on import.
"""

from .bands import active_bands_for_pilot, assign_band, expand_band_allowed
from .config import PilotConfig
from .label_classes import LabelClass

__all__ = ["LabelClass", "PilotConfig", "active_bands_for_pilot", "assign_band", "expand_band_allowed"]
