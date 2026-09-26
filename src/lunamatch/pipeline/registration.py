"""SIFT → descriptor matching → RANSAC → image registration."""

from time import perf_counter
from dataclasses import asdict

import cv2
import numpy as np

from lunamatch.distribution.coverage import coverage_metrics
from lunamatch.distribution.spatial_selection import select_uniform
from lunamatch.evaluation.metrics import registration_metrics
from lunamatch.evaluation.visualization import match_visualization
from lunamatch.features.sift import extract_sift
from lunamatch.features.orb import extract_orb
from lunamatch.features.akaze import extract_akaze
from lunamatch.features.learned_descriptor import extract_learned
from lunamatch.geometry.ransac import GeometryError, estimate_transform
from lunamatch.geometry.warping import warp_source
from lunamatch.ingestion.metadata import LunarImage
from lunamatch.matching.classical import match_features
from lunamatch.matching.loftr import match_loftr
from lunamatch.matching.lightglue import match_lightglue
from lunamatch.matching.cross_modal import combine_correspondences
from lunamatch.matching.matcher_interface import Correspondences
from lunamatch.preprocessing.normalization import matching_gray
from lunamatch.preprocessing.illumination import local_normalize
from lunamatch.preprocessing.gradients import gradient_magnitude
from lunamatch.preprocessing.shadow import shadow_mask
from lunamatch.preprocessing.pyramid import build_pyramid
from lunamatch.preprocessing.cross_modal import iirs_pca_plane
from lunamatch.refinement.subpixel import refine_points
from lunamatch.geometry.ransac import transform_points

from .config import RegistrationConfig
from .result import RegistrationResult


def _classical_multiscale(src: np.ndarray, ref: np.ndarray,
                          src_mask: np.ndarray, ref_mask: np.ndarray,
                          config: RegistrationConfig) -> tuple[Correspondences, int, int]:
    """Try all configured pyramid-level pairs and keep most RANSAC inliers."""
    extractor = {"sift": extract_sift, "orb": extract_orb, "akaze": extract_akaze,
                 "descriptor": extract_learned}[config.matcher]
    if config.matcher == "descriptor" and config.descriptor_checkpoint:
        extractor = lambda image, mask, max_features: extract_learned(
            image, mask, max_features, checkpoint=config.descriptor_checkpoint
        )
    src_levels = build_pyramid(src, config.pyramid_levels)
    ref_levels = build_pyramid(ref, config.pyramid_levels)
    source_features = []
    reference_features = []
    for image, _, _ in src_levels:
        mask = cv2.resize(src_mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        source_features.append(extractor(image, mask, config.max_features))
    for image, _, _ in ref_levels:
        mask = cv2.resize(ref_mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
        reference_features.append(extractor(image, mask, config.max_features))
    best = None
    best_score = -1
    for i, source_feature in enumerate(source_features):
        for j, reference_feature in enumerate(reference_features):
            candidates = match_features(source_feature, reference_feature, config.ratio)
            if len(candidates) < (4 if config.geometry_model == "homography" else 3):
                continue
            source_scale = np.array(src_levels[i][1:], dtype=np.float32)
            reference_scale = np.array(ref_levels[j][1:], dtype=np.float32)
            points = Correspondences(candidates.source_points * source_scale,
                                     candidates.reference_points * reference_scale,
                                     candidates.confidence)
            try:
                geometry = estimate_transform(points, model=config.geometry_model,
                                              threshold=config.ransac_threshold,
                                              confidence=config.ransac_confidence)
            except GeometryError:
                continue
            score = int(geometry.inliers.sum())
            if score > best_score:
                best = (points, len(source_feature.keypoints), len(reference_feature.keypoints))
                best_score = score
    if best is None:
        raise GeometryError("No pyramid-level pair produced a valid transformation")
    return best


def register(source: LunarImage, reference: LunarImage,
             config: RegistrationConfig | None = None) -> RegistrationResult:
    """Register two images with the working classical baseline."""
    started = perf_counter()
    config = config or RegistrationConfig()
    extractors = {"sift": extract_sift, "orb": extract_orb, "akaze": extract_akaze,
                  "descriptor": extract_learned}
    if config.matcher not in (*extractors, "loftr", "lightglue", "hybrid"):
        raise ValueError(f"Matcher is unavailable in this milestone: {config.matcher}")
    if config.iirs_mode not in ("pca", "band"):
        raise ValueError(f"Unsupported IIRS representation: {config.iirs_mode}")
    source_rep = iirs_pca_plane(source) if source.sensor == "IIRS" and config.iirs_mode == "pca" and source.bands > 1 else source
    reference_rep = iirs_pca_plane(reference) if reference.sensor == "IIRS" and config.iirs_mode == "pca" and reference.bands > 1 else reference
    src = matching_gray(source_rep, band=config.source_band, clahe=config.clahe)
    ref = matching_gray(reference_rep, band=config.reference_band, clahe=config.clahe)
    if config.local_normalization:
        src, ref = local_normalize(src), local_normalize(ref)
    if config.gradient:
        src, ref = gradient_magnitude(src), gradient_magnitude(ref)
    src_mask = source_rep.valid_mask if source_rep.valid_mask is not None else np.ones(src.shape, bool)
    ref_mask = reference_rep.valid_mask if reference_rep.valid_mask is not None else np.ones(ref.shape, bool)
    if config.shadow_mask:
        src_mask = shadow_mask(src, src_mask)
        ref_mask = shadow_mask(ref, ref_mask)
    if config.matcher == "loftr":
        matches, device = match_loftr(src, ref)
        source_keypoints = reference_keypoints = None
    elif config.matcher == "lightglue":
        matches, device, source_keypoints, reference_keypoints = match_lightglue(
            src, ref, max_keypoints=min(config.max_features, 2048))
    elif config.matcher == "hybrid":
        classical_config = RegistrationConfig(
            matcher="sift", geometry_model=config.geometry_model,
            ransac_threshold=config.ransac_threshold,
            ransac_confidence=config.ransac_confidence,
            ratio=config.ratio, max_features=config.max_features,
            pyramid_levels=config.pyramid_levels)
        classical, source_keypoints, reference_keypoints = _classical_multiscale(
            src, ref, src_mask, ref_mask, classical_config)
        learned, device = match_loftr(src, ref)
        matches = combine_correspondences(classical, learned)
    else:
        matches, source_keypoints, reference_keypoints = _classical_multiscale(
            src, ref, src_mask, ref_mask, config)
        device = "cpu"
    try:
        geometry = estimate_transform(matches, model=config.geometry_model,
                                      threshold=config.ransac_threshold,
                                      confidence=config.ransac_confidence)
    except GeometryError as exc:
        raise GeometryError(f"Registration failed after {len(matches)} candidate matches: {exc}") from exc
    registered_preview = warp_source(src, geometry.matrix, ref.shape)
    registered = warp_source(np.asarray(source.image), geometry.matrix, ref.shape)
    metrics = registration_metrics(geometry, source_keypoints=source_keypoints,
                                   reference_keypoints=reference_keypoints,
                                   candidate_matches=len(matches),
                                   runtime_seconds=perf_counter() - started)
    metrics.update(coverage_metrics(matches.reference_points, geometry.inliers, ref.shape))
    if config.spatial_selection:
        selected = select_uniform(matches.reference_points, matches.confidence,
                                  geometry.inliers, ref.shape, rows=config.grid_rows,
                                  cols=config.grid_cols, max_per_cell=config.max_matches_per_cell)
    else:
        selected = np.flatnonzero(geometry.inliers)
    chosen = np.zeros(len(matches), dtype=bool)
    chosen[selected] = True
    if config.subpixel_refinement and len(selected):
        refined, success = refine_points(src, ref, matches.source_points[selected],
                                         matches.reference_points[selected],
                                         window_size=config.refinement_window_size,
                                         search_radius=config.refinement_search_radius)
        matches.reference_points[selected] = refined
        geometry.residuals[selected] = np.linalg.norm(
            transform_points(matches.source_points[selected], geometry.matrix) - refined,
            axis=1)
        metrics["estimated_subpixel_coordinates"] = int(success.sum())
        metrics["refined_selected_residual_rmse_px"] = float(np.sqrt(np.mean(
            geometry.residuals[selected] ** 2)))
        metrics["subpixel_accuracy_validated"] = False
    metrics["selected_correspondences"] = len(selected)
    metrics["selected_coverage"] = coverage_metrics(matches.reference_points,
                                                      chosen, ref.shape,
                                                      config.grid_rows,
                                                      config.grid_cols)["spatial_coverage"]
    metrics.update({"matcher": config.matcher, "device": device})
    metrics["metadata_scale_ratio"] = (
        source.pixel_scale / reference.pixel_scale
        if source.pixel_scale is not None and reference.pixel_scale is not None
        and reference.pixel_scale > 0 else None)
    visualization = match_visualization(src, ref, matches.source_points,
                                        matches.reference_points, chosen)
    provenance = {
        "source": source.summary(),
        "reference": reference.summary(),
        "configuration": asdict(config),
        "model": ("Kornia LoFTR outdoor" if config.matcher in ("loftr", "hybrid")
                  else "CVG SuperPoint + LightGlue" if config.matcher == "lightglue" else config.matcher),
        "device": device,
    }
    return RegistrationResult(src, ref, registered, registered_preview, matches, geometry, metrics,
                              visualization, selected, provenance)
