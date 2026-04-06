# Authors: see git history
#
# Copyright (c) 2026 Authors
# Licensed under the GNU GPL version 3.0 or later.  See the file LICENSE for details.

"""Utilities for tracing the outline of a raster image into a polygon path.

This module is intentionally free of inkstitch-specific imports so it can be
tested independently.  It operates exclusively on PIL Images and returns plain
Python lists of (x, y) float tuples in image-pixel coordinates.

Supported image types:
  - PNG with meaningful transparency (alpha-based mode)
  - JPEG / opaque PNG / any image without alpha (luminance-based mode)
"""

from collections import deque

import numpy as np
from PIL import Image, ImageFilter
from shapely.geometry import MultiPoint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_outline_coords(pil_image, mode='auto', threshold_adjust=0, simplification=1.0):
    """Return the outer contour of *pil_image* as a list of (x, y) pixel tuples.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Source image (any mode; will be converted internally).
    mode : str
        ``'auto'``       – detect mode from alpha channel presence.
        ``'alpha'``      – use alpha channel; best for transparent PNGs.
        ``'luminance'``  – use luminance + Otsu; best for photos/JPEGs.
    threshold_adjust : int
        Offset added to the auto-computed threshold (range −50 … +50).
        Positive values include more pixels (loosen the threshold).
    simplification : float
        Shapely ``simplify`` tolerance in *pixels*.  Higher = fewer vertices.

    Returns
    -------
    list[tuple[float, float]] or None
        Exterior coordinates of the largest detected region, or ``None`` if no
        foreground region could be found.
    """
    img_rgba = pil_image.convert('RGBA')
    orig_w, orig_h = img_rgba.size

    # Choose detection mode
    if mode == 'auto':
        use_alpha = has_meaningful_alpha(img_rgba)
    elif mode == 'alpha':
        use_alpha = True
    else:  # 'luminance'
        use_alpha = False

    if use_alpha:
        mask_pil = extract_mask_alpha(img_rgba, threshold_adjust)
    else:
        mask_pil = extract_mask_luminance(img_rgba, threshold_adjust)

    if mask_pil is None:
        return None

    # Work at reduced resolution for contour extraction
    max_dim = 500
    scale = min(1.0, max_dim / max(orig_w, orig_h))
    small_w = max(1, int(orig_w * scale))
    small_h = max(1, int(orig_h * scale))

    small_mask = mask_pil.resize((small_w, small_h), Image.NEAREST)

    coords = mask_to_contour(small_mask, simplification * scale)
    if coords is None:
        return None

    # Scale coordinates back to original pixel space and clamp to image bounds
    inv_scale = 1.0 / scale
    return [
        (max(0.0, min(orig_w, x * inv_scale)),
         max(0.0, min(orig_h, y * inv_scale)))
        for x, y in coords
    ]


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def has_meaningful_alpha(img_rgba):
    """Return True when more than 5 % of pixels have alpha < 250."""
    alpha = np.array(img_rgba.getchannel('A'), dtype=np.uint8)
    transparent_ratio = np.sum(alpha < 250) / alpha.size
    return bool(transparent_ratio > 0.05)


# ---------------------------------------------------------------------------
# Mask extraction – alpha mode
# ---------------------------------------------------------------------------

def extract_mask_alpha(img_rgba, threshold_adjust=0):
    """Build a binary mask from the alpha channel."""
    alpha = img_rgba.getchannel('A')

    # Blur to smooth jagged anti-aliased edges
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=2))

    raw_threshold = max(0, min(255, 128 + threshold_adjust * 255 // 100))
    binary = alpha.point(lambda v: 255 if v > raw_threshold else 0)

    return _morphological_cleanup(binary)


# ---------------------------------------------------------------------------
# Mask extraction – luminance mode (real photos / JPEGs)
# ---------------------------------------------------------------------------

def extract_mask_luminance(img_rgba, threshold_adjust=0):
    """Build a binary mask using Otsu thresholding on the luminance channel.

    Works for fully-opaque images such as JPEGs where an alpha-based approach
    would either include everything or nothing.
    """
    # Convert to greyscale for luminance-based analysis
    gray = Image.merge('RGB', img_rgba.split()[:3]).convert('L')
    w, h = gray.size

    # Heavy blur to destroy texture / noise while keeping large-scale edges
    blur_radius = max(5, min(w, h) // 50)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(blurred, dtype=np.uint8)

    # Otsu auto-threshold
    t = otsu_threshold(arr) + threshold_adjust
    t = max(0, min(255, t))

    # Binary mask
    mask_arr = (arr > t).astype(np.uint8) * 255

    # Polarity: sample the image border – it is almost always background.
    border_w = max(1, min(w, h) // 20)
    border_mean = _border_mean(arr, border_w)
    if border_mean > t:
        # Background is bright → subject is dark → invert
        mask_arr = 255 - mask_arr

    mask_pil = Image.fromarray(mask_arr, mode='L')
    mask_pil = _morphological_cleanup(mask_pil)

    # Keep only the largest connected region (removes background islands)
    mask_pil = _keep_largest_region(mask_pil)

    # Final smooth pass to ease contour extraction
    mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=3))
    mask_pil = mask_pil.point(lambda v: 255 if v > 127 else 0)

    return mask_pil


# ---------------------------------------------------------------------------
# Contour extraction
# ---------------------------------------------------------------------------

def mask_to_contour(mask_pil, simplification_tolerance=1.0):
    """Convert a binary PIL mask to an ordered exterior contour.

    Parameters
    ----------
    mask_pil : PIL.Image.Image (mode 'L')
        Binary mask (0 = background, 255 = foreground).
    simplification_tolerance : float
        Shapely simplify tolerance in pixels.

    Returns
    -------
    list[tuple[float, float]] or None
    """
    w, h = mask_pil.size

    # Detect boundary pixels (foreground pixel adjacent to at least one bg pixel)
    mask_arr = np.array(mask_pil, dtype=np.uint8) // 255  # 0 or 1
    padded = np.pad(mask_arr, 1, mode='constant', constant_values=0)
    boundary = (
        mask_arr.astype(bool) & (
            (padded[:-2, 1:-1] == 0)
            | (padded[2:, 1:-1] == 0)
            | (padded[1:-1, :-2] == 0)
            | (padded[1:-1, 2:] == 0)
        )
    )

    ys, xs = np.where(boundary)
    if len(xs) == 0:
        return None

    points = list(zip(xs.astype(float), ys.astype(float)))

    # Convert point cloud → polygon using buffer trick (no scipy / cv2 needed)
    mp = MultiPoint(points)
    buf_size = max(1.5, simplification_tolerance)
    polygon = mp.buffer(buf_size).buffer(-buf_size * 0.4)

    if polygon.is_empty:
        return None

    # Keep only the largest polygon when result is MultiPolygon
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda g: g.area)

    if polygon.geom_type != 'Polygon':
        return None

    # Simplify to reduce vertex count
    tol = max(1.0, simplification_tolerance)
    simplified = polygon.simplify(tol, preserve_topology=True)

    if simplified.is_empty or simplified.geom_type not in ('Polygon', 'MultiPolygon'):
        simplified = polygon  # fall back to un-simplified

    if simplified.geom_type == 'MultiPolygon':
        simplified = max(simplified.geoms, key=lambda g: g.area)

    return list(simplified.exterior.coords)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def otsu_threshold(arr):
    """Compute Otsu's optimal threshold for a uint8 numpy array.

    Returns the integer threshold value that maximises between-class variance.
    """
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))

    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_t = 0

    for t in range(256):
        weight_bg += int(hist[t])
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * float(hist[t])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = float(weight_bg) * float(weight_fg) * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_t = t

    return best_t


def _border_mean(arr, border_width):
    """Return the mean pixel value of the border strip of *arr*."""
    h, w = arr.shape
    bw = min(border_width, h // 2, w // 2)
    top = arr[:bw, :]
    bottom = arr[-bw:, :]
    left = arr[:, :bw]
    right = arr[:, -bw:]
    border_pixels = np.concatenate([top.ravel(), bottom.ravel(),
                                    left.ravel(), right.ravel()])
    return float(border_pixels.mean()) if len(border_pixels) else 128.0


def _morphological_cleanup(mask_pil):
    """Apply morphological close then open to fill gaps and remove noise."""
    # Close: dilate then erode
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(5))
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(5))
    # Open: erode then dilate
    mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    return mask_pil


def _keep_largest_region(mask_pil):
    """Keep only the largest connected foreground region using BFS.

    Operates on a downscaled copy for performance, then upscales the result.
    """
    orig_w, orig_h = mask_pil.size

    # Downscale for BFS performance
    scale = min(1.0, 200 / max(orig_w, orig_h))
    small_w = max(1, int(orig_w * scale))
    small_h = max(1, int(orig_h * scale))
    small = mask_pil.resize((small_w, small_h), Image.NEAREST)

    arr = (np.array(small, dtype=np.uint8) > 127).astype(np.int32)

    # BFS to label connected components
    labels = np.zeros_like(arr, dtype=np.int32)
    current_label = 0
    label_sizes = {}

    for start_y in range(small_h):
        for start_x in range(small_w):
            if arr[start_y, start_x] == 1 and labels[start_y, start_x] == 0:
                current_label += 1
                size = 0
                queue = deque()
                queue.append((start_x, start_y))
                labels[start_y, start_x] = current_label
                while queue:
                    cx, cy = queue.popleft()
                    size += 1
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy),
                                   (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < small_w and 0 <= ny < small_h:
                            if arr[ny, nx] == 1 and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                queue.append((nx, ny))
                label_sizes[current_label] = size

    if not label_sizes:
        return mask_pil  # nothing found; return as-is

    best_label = max(label_sizes, key=lambda k: label_sizes[k])
    result_arr = ((labels == best_label).astype(np.uint8) * 255)

    result_small = Image.fromarray(result_arr, mode='L')
    # Upscale back to original size
    return result_small.resize((orig_w, orig_h), Image.NEAREST)
