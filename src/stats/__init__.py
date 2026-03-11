"""
Post-experiment statistics: subject-level variability, backbone interaction,
and cross-dataset consistency (no pooling).
"""

from .variability import compute_variability
from .backbone_interaction import compute_backbone_interaction
from .cross_dataset import write_cross_dataset_report

__all__ = [
    "compute_variability",
    "compute_backbone_interaction",
    "write_cross_dataset_report",
]
