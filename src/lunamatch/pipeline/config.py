"""Typed configuration for the verified SIFT baseline."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class RegistrationConfig:
    """Parameters for the current SIFT baseline."""

    matcher: str = "sift"
    descriptor_checkpoint: str | None = None
    geometry_model: str = "homography"
    ransac_threshold: float = 3.0
    ransac_confidence: float = 0.99
    ratio: float = 0.75
    max_features: int = 6000
    clahe: bool = False
    local_normalization: bool = False
    gradient: bool = False
    shadow_mask: bool = False
    pyramid_levels: int = 1
    spatial_selection: bool = True
    grid_rows: int = 8
    grid_cols: int = 8
    max_matches_per_cell: int = 10
    subpixel_refinement: bool = False
    refinement_window_size: int = 11
    refinement_search_radius: int = 3
    iirs_mode: str = "pca"
    source_band: int = 0
    reference_band: int = 0

    def __post_init__(self) -> None:
        """Reject parameter combinations that cannot be run meaningfully."""
        if self.matcher not in ("sift", "orb", "akaze", "descriptor", "loftr", "lightglue", "hybrid"):
            raise ValueError(f"Unsupported matcher: {self.matcher}")
        if self.geometry_model not in ("affine", "homography"):
            raise ValueError(f"Unsupported geometry model: {self.geometry_model}")
        if not 0 < self.ratio < 1 or self.max_features < 10:
            raise ValueError("Matcher ratio must be in (0,1) and max_features >=10")
        if not 1 <= self.pyramid_levels <= 5:
            raise ValueError("pyramid_levels must be 1..5")
        if min(self.grid_rows, self.grid_cols, self.max_matches_per_cell) < 1:
            raise ValueError("Distribution grid settings must be positive")
        if self.iirs_mode not in ("pca", "band"):
            raise ValueError(f"Unsupported IIRS mode: {self.iirs_mode}")


def load_config(path: str | Path) -> RegistrationConfig:
    """Load supported registration parameters from YAML, rejecting unknown keys."""
    with Path(path).open() as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    allowed_sections = {"matcher", "preprocessing", "geometry", "distribution", "refinement", "cross_modal"}
    unknown = set(data) - allowed_sections
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    values = {}
    fields = {
        "matcher": {"name": "matcher", "ratio": "ratio", "max_features": "max_features",
                    "descriptor_checkpoint": "descriptor_checkpoint"},
        "preprocessing": {"clahe": "clahe", "local_normalization": "local_normalization",
                          "gradient": "gradient", "shadow_mask": "shadow_mask",
                          "pyramid_levels": "pyramid_levels", "source_band": "source_band",
                          "reference_band": "reference_band"},
        "geometry": {"model": "geometry_model", "ransac_threshold": "ransac_threshold",
                     "confidence": "ransac_confidence"},
        "distribution": {"enabled": "spatial_selection", "grid_rows": "grid_rows",
                         "grid_cols": "grid_cols", "max_matches_per_cell": "max_matches_per_cell"},
        "refinement": {"enabled": "subpixel_refinement", "window_size": "refinement_window_size",
                       "search_radius": "refinement_search_radius"},
        "cross_modal": {"iirs_mode": "iirs_mode"},
    }
    for section, mapping in fields.items():
        contents = data.get(section, {})
        if not isinstance(contents, dict):
            raise ValueError(f"Configuration section {section} must be a mapping")
        extra = set(contents) - set(mapping)
        if extra:
            raise ValueError(f"Unknown {section} settings: {sorted(extra)}")
        values.update({mapping[key]: value for key, value in contents.items()})
    return RegistrationConfig(**values)
