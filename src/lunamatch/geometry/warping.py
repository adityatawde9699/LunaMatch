"""Warp source image into reference pixel coordinates."""

import cv2
import numpy as np

_MAX_WARP_AXIS = 16000


def _warp_tiled(source: np.ndarray, matrix: np.ndarray,
                height: int, width: int) -> np.ndarray:
    """Warp large rasters in bounded source and destination tiles."""
    output = np.zeros((height, width), dtype=source.dtype)
    matrix = np.asarray(matrix, dtype=np.float64)
    source_height, source_width = source.shape
    # Adjacent source tiles overlap; one pixel at each internal edge is
    # excluded from its validity mask so interpolation uses real neighbors.
    tile_step = _MAX_WARP_AXIS - 4
    for sy in range(0, source_height, tile_step):
        for sx in range(0, source_width, tile_step):
            source_tile = source[sy:min(sy + _MAX_WARP_AXIS, source_height),
                                 sx:min(sx + _MAX_WARP_AXIS, source_width)]
            tile_height, tile_width = source_tile.shape
            corners = np.array([[[0, 0], [tile_width - 1, 0],
                                 [tile_width - 1, tile_height - 1],
                                 [0, tile_height - 1]]], dtype=np.float64)
            source_offset = np.array([[1, 0, sx], [0, 1, sy], [0, 0, 1]],
                                     dtype=np.float64)
            mapped = cv2.perspectiveTransform(corners, matrix @ source_offset)[0]
            if not np.isfinite(mapped).all():
                continue
            min_x = max(0, int(np.floor(mapped[:, 0].min())) - 2)
            max_x = min(width, int(np.ceil(mapped[:, 0].max())) + 3)
            min_y = max(0, int(np.floor(mapped[:, 1].min())) - 2)
            max_y = min(height, int(np.ceil(mapped[:, 1].max())) + 3)
            if min_x >= max_x or min_y >= max_y:
                continue

            source_mask = np.full(source_tile.shape, 255, dtype=np.uint8)
            if sx > 0:
                source_mask[:, 0] = 0
            if sx + tile_width < source_width:
                source_mask[:, -1] = 0
            if sy > 0:
                source_mask[0, :] = 0
            if sy + tile_height < source_height:
                source_mask[-1, :] = 0
            for dy in range(min_y, max_y, _MAX_WARP_AXIS):
                end_y = min(dy + _MAX_WARP_AXIS, max_y)
                for dx in range(min_x, max_x, _MAX_WARP_AXIS):
                    end_x = min(dx + _MAX_WARP_AXIS, max_x)
                    destination_offset = np.array(
                        [[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
                    local_matrix = destination_offset @ matrix @ source_offset
                    size = (end_x - dx, end_y - dy)
                    warped = cv2.warpPerspective(
                        source_tile, local_matrix, size,
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    valid = cv2.warpPerspective(
                        source_mask, local_matrix, size,
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                    target = output[dy:end_y, dx:end_x]
                    target[valid > 0] = warped[valid > 0]
    return output


def warp_source(source: np.ndarray, matrix: np.ndarray,
                reference_shape: tuple[int, int]) -> np.ndarray:
    """Warp a source plane or band cube to reference height and width."""
    height, width = reference_shape
    if source.ndim == 3:
        return np.stack([warp_source(source[:, :, band], matrix, reference_shape)
                         for band in range(source.shape[2])], axis=-1)
    if source.dtype not in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        raise ValueError(f"Warping does not support pixel dtype {source.dtype}")
    source_height, source_width = source.shape
    if max(height, width, source_height, source_width) >= 32767:
        return _warp_tiled(source, matrix, height, width)
    return cv2.warpPerspective(source, matrix, (width, height),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
