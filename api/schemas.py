"""Validated API inputs and registration summary output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lunamatch.pipeline.config import RegistrationConfig


class RegistrationOptions(BaseModel):
    """Client-selectable image processing parameters."""

    model_config = ConfigDict(extra="forbid")
    clahe: bool = False
    local_normalization: bool = False
    gradient: bool = False
    shadow_mask: bool = False
    pyramid_levels: int = Field(default=1, ge=1, le=5)
    geometry_model: Literal["affine", "homography"] = "homography"
    spatial_selection: bool = True
    grid_rows: int = Field(default=8, ge=1, le=32)
    grid_cols: int = Field(default=8, ge=1, le=32)
    max_matches_per_cell: int = Field(default=10, ge=1, le=100)
    subpixel_refinement: bool = False
    source_band: int = Field(default=0, ge=0)
    reference_band: int = Field(default=0, ge=0)
    iirs_mode: Literal["pca", "band"] = "pca"

    def to_config(self, matcher: str) -> RegistrationConfig:
        """Convert an API request into pipeline configuration."""
        return RegistrationConfig(matcher=matcher, **self.model_dump())
