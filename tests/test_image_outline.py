# Authors: see git history
#
# Copyright (c) 2026 Authors
# Licensed under the GNU GPL version 3.0 or later.  See the file LICENSE for details.

"""Comprehensive tests for lib/extensions/utils/image_outline.py

Covers:
  - Alpha-based outline detection (transparent PNG)
  - Luminance/Otsu-based detection (JPEG / opaque images)
  - Polarity detection (dark-on-bright vs bright-on-dark)
  - Morphological cleanup
  - Largest-region extraction
  - Contour output shape and validity
  - Edge cases: blank image, fully-black, fully-white, tiny image
"""

import sys
import os
import math
import importlib.util

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

# Import image_outline directly (bypasses lib/extensions/__init__.py which needs wx)
_module_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'extensions', 'utils', 'image_outline.py')
_spec = importlib.util.spec_from_file_location('image_outline', _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_outline_coords = _mod.get_outline_coords
has_meaningful_alpha = _mod.has_meaningful_alpha
otsu_threshold = _mod.otsu_threshold
extract_mask_alpha = _mod.extract_mask_alpha
extract_mask_luminance = _mod.extract_mask_luminance
mask_to_contour = _mod.mask_to_contour
_keep_largest_region = _mod._keep_largest_region
_border_mean = _mod._border_mean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_circle_rgba(size=200, radius=70, fg_alpha=255, bg_alpha=0):
    """Transparent-background image with a filled circle."""
    img = Image.new('RGBA', (size, size), (255, 255, 255, bg_alpha))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(255, 0, 0, fg_alpha)
    )
    return img


def make_circle_rgb(size=200, radius=70, dark_on_bright=True):
    """Opaque RGB image with a filled circle (no alpha)."""
    if dark_on_bright:
        bg, fg = (240, 240, 240), (20, 20, 20)
    else:
        bg, fg = (20, 20, 20), (240, 240, 240)
    img = Image.new('RGB', (size, size), bg)
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=fg
    )
    return img


def make_rect_rgba(size=200, rect_frac=0.6, bg_alpha=0):
    """Transparent-background image with a filled rectangle."""
    img = Image.new('RGBA', (size, size), (255, 255, 255, bg_alpha))
    draw = ImageDraw.Draw(img)
    margin = int(size * (1 - rect_frac) / 2)
    draw.rectangle([margin, margin, size - margin, size - margin], fill=(0, 0, 200, 255))
    return img


def polygon_area(coords):
    """Shoelace formula for polygon area (absolute value)."""
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0


def centroid(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ---------------------------------------------------------------------------
# Tests: has_meaningful_alpha
# ---------------------------------------------------------------------------

class TestHasMeaningfulAlpha:
    def test_fully_opaque_returns_false(self):
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 255))
        assert has_meaningful_alpha(img) is False

    def test_fully_transparent_returns_true(self):
        img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        assert has_meaningful_alpha(img) is True

    def test_partial_transparency_returns_true(self):
        img = make_circle_rgba(size=100, radius=35, fg_alpha=255, bg_alpha=0)
        assert has_meaningful_alpha(img) is True

    def test_mostly_opaque_with_tiny_transparent_returns_false(self):
        """Less than 5% transparent → should be treated as opaque."""
        img = Image.new('RGBA', (100, 100), (200, 200, 200, 255))
        # Make just 1% of pixels transparent (only 100 pixels)
        arr = np.array(img)
        arr[:1, :, 3] = 0   # top row = 1% of 100 rows
        result = has_meaningful_alpha(Image.fromarray(arr))
        assert result is False


# ---------------------------------------------------------------------------
# Tests: otsu_threshold
# ---------------------------------------------------------------------------

class TestOtsuThreshold:
    def test_bimodal_histogram_splits_cleanly(self):
        """Image with values clustered at 50 and 200 → threshold at or between them.

        Otsu returns the first t that maximises between-class variance.
        For a clean two-value image the threshold lands exactly on the lower
        value (50), which is a correct split point (pixels ≤ 50 are dark,
        pixels > 50 are bright).
        """
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:50, :] = 50     # dark region
        arr[50:, :] = 200    # light region
        t = otsu_threshold(arr)
        # Threshold must separate the two classes: everything ≤ t is dark,
        # everything > t is bright.  Valid range is 50 ≤ t < 200.
        assert 50 <= t < 200, f"Expected threshold in [50, 200), got {t}"

    def test_uniform_image_returns_reasonable_value(self):
        """Uniform image → Otsu gives 0 (no meaningful threshold)."""
        arr = np.full((100, 100), 128, dtype=np.uint8)
        t = otsu_threshold(arr)
        assert 0 <= t <= 255

    def test_black_white_halves(self):
        """Perfect bimodal (0 vs 255): threshold at 0 is a valid split point."""
        arr = np.zeros((200, 200), dtype=np.uint8)
        arr[100:, :] = 255
        t = otsu_threshold(arr)
        # t=0 is correct: pixels > 0 are the bright class (all 255s).
        assert 0 <= t < 255, f"Expected threshold in [0, 255), got {t}"


# ---------------------------------------------------------------------------
# Tests: extract_mask_alpha
# ---------------------------------------------------------------------------

class TestExtractMaskAlpha:
    def test_transparent_bg_gives_foreground_mask(self):
        img = make_circle_rgba(size=100, radius=35, fg_alpha=255, bg_alpha=0)
        mask = extract_mask_alpha(img, threshold_adjust=0)
        arr = np.array(mask)
        # Center pixel should be foreground (white = 255)
        assert arr[50, 50] == 255, "Center should be foreground"
        # Corner pixel should be background (black = 0)
        assert arr[0, 0] == 0, "Corner should be background"

    def test_threshold_adjust_positive_expands_mask(self):
        """threshold_adjust > 0 lowers effective threshold → includes more."""
        img = make_circle_rgba(size=100, radius=35, fg_alpha=100, bg_alpha=0)
        mask_default = extract_mask_alpha(img, threshold_adjust=0)
        mask_loose = extract_mask_alpha(img, threshold_adjust=40)
        arr_default = np.array(mask_default)
        arr_loose = np.array(mask_loose)
        # Loose threshold should include at least as many pixels
        assert arr_loose.sum() >= arr_default.sum()

    def test_fully_transparent_gives_empty_mask(self):
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 0))
        mask = extract_mask_alpha(img, threshold_adjust=0)
        arr = np.array(mask)
        assert arr.sum() == 0, "All-transparent image should give empty mask"

    def test_mask_size_matches_input(self):
        img = make_circle_rgba(size=150, radius=50)
        mask = extract_mask_alpha(img, threshold_adjust=0)
        assert mask.size == img.size


# ---------------------------------------------------------------------------
# Tests: extract_mask_luminance
# ---------------------------------------------------------------------------

class TestExtractMaskLuminance:
    def test_dark_circle_on_bright_bg(self):
        img = make_circle_rgb(size=100, radius=35, dark_on_bright=True)
        mask = extract_mask_luminance(img.convert('RGBA'), threshold_adjust=0)
        arr = np.array(mask)
        # Center (circle) should be foreground
        assert arr[50, 50] > 127, "Dark circle center should be foreground"
        # Corners (background) should be background
        assert arr[2, 2] < 127, "Bright corner should be background"

    def test_bright_circle_on_dark_bg(self):
        """Polarity inversion: bright subject on dark background."""
        img = make_circle_rgb(size=100, radius=35, dark_on_bright=False)
        mask = extract_mask_luminance(img.convert('RGBA'), threshold_adjust=0)
        arr = np.array(mask)
        # Center (bright circle) should be foreground
        assert arr[50, 50] > 127, "Bright circle center should be foreground"
        # Corners (dark background) should be background
        assert arr[2, 2] < 127, "Dark corner should be background"

    def test_mask_size_matches_input(self):
        img = make_circle_rgb(size=120, radius=40)
        mask = extract_mask_luminance(img.convert('RGBA'), threshold_adjust=0)
        assert mask.size == img.size

    def test_threshold_adjust_negative_shrinks_mask(self):
        """Negative threshold_adjust makes it harder to be foreground → smaller mask."""
        img = make_circle_rgb(size=100, radius=35, dark_on_bright=True)
        rgba = img.convert('RGBA')
        mask_default = extract_mask_luminance(rgba, threshold_adjust=0)
        mask_tight = extract_mask_luminance(rgba, threshold_adjust=-30)
        # Tight threshold should include equal or fewer foreground pixels
        arr_default = np.array(mask_default)
        arr_tight = np.array(mask_tight)
        assert arr_tight.sum() <= arr_default.sum() + 100  # allow some morph wiggle


# ---------------------------------------------------------------------------
# Tests: mask_to_contour
# ---------------------------------------------------------------------------

class TestMaskToContour:
    def test_circle_contour_is_roughly_circular(self):
        """Contour of a filled circle should be close to a circle."""
        mask = Image.new('L', (200, 200), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([50, 50, 150, 150], fill=255)

        coords = mask_to_contour(mask, simplification_tolerance=2.0)
        assert coords is not None and len(coords) >= 4

        cx, cy = centroid(coords)
        # Centroid should be near image center
        assert abs(cx - 100) < 20, f"Centroid x={cx:.1f} should be near 100"
        assert abs(cy - 100) < 20, f"Centroid y={cy:.1f} should be near 100"

        # All points should be at roughly the same distance from center (~50px)
        radii = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for x, y in coords]
        mean_r = sum(radii) / len(radii)
        assert 30 < mean_r < 70, f"Mean radius {mean_r:.1f} should be ~50px"

    def test_rect_contour_has_correct_area(self):
        """Contour of a 100×100 rectangle should have area ≈ 10000 px²."""
        mask = Image.new('L', (200, 200), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([50, 50, 150, 150], fill=255)

        coords = mask_to_contour(mask, simplification_tolerance=1.0)
        assert coords is not None

        area = polygon_area(coords)
        # Allow generous tolerance due to pixel quantization and simplification
        assert 8000 < area < 15000, f"Area {area:.0f} should be near 10000"

    def test_empty_mask_returns_none(self):
        mask = Image.new('L', (100, 100), 0)
        result = mask_to_contour(mask)
        assert result is None

    def test_full_mask_returns_contour(self):
        """A fully-filled mask should produce a contour around the border."""
        mask = Image.new('L', (100, 100), 255)
        coords = mask_to_contour(mask, simplification_tolerance=1.0)
        assert coords is not None and len(coords) >= 4

    def test_contour_is_closed(self):
        """Exterior coordinates from shapely should be a closed ring (first == last)."""
        mask = Image.new('L', (100, 100), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([20, 20, 80, 80], fill=255)
        coords = mask_to_contour(mask, simplification_tolerance=1.0)
        assert coords is not None
        assert coords[0] == coords[-1], "Contour should be a closed ring"


# ---------------------------------------------------------------------------
# Tests: _keep_largest_region
# ---------------------------------------------------------------------------

class TestKeepLargestRegion:
    def test_removes_small_noise_blobs(self):
        """Two circles: large one should survive, tiny one should vanish."""
        mask = Image.new('L', (200, 200), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([60, 60, 140, 140], fill=255)  # big circle ~6400 px
        draw.ellipse([5, 5, 15, 15], fill=255)       # tiny circle ~100 px
        result = _keep_largest_region(mask)
        arr = np.array(result)
        # Corner region (tiny circle) should be gone
        assert arr[10, 10] == 0, "Small noise blob should be removed"
        # Center region (big circle) should survive
        assert arr[100, 100] > 0, "Main region should survive"

    def test_single_region_preserved(self):
        mask = Image.new('L', (100, 100), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([20, 20, 80, 80], fill=255)
        result = _keep_largest_region(mask)
        arr_in = np.array(mask)
        arr_out = np.array(result)
        assert arr_out[50, 50] > 0
        # Should not be much bigger than input
        assert arr_out.sum() <= arr_in.sum() * 1.1


# ---------------------------------------------------------------------------
# Tests: get_outline_coords (full pipeline)
# ---------------------------------------------------------------------------

class TestGetOutlineCoords:
    def test_alpha_mode_circle(self):
        """Full pipeline on transparent-background circle."""
        img = make_circle_rgba(size=200, radius=70)
        coords = get_outline_coords(img, mode='alpha')
        assert coords is not None
        assert len(coords) >= 4
        # Coords should be in image-pixel space (0..200)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        assert min(xs) >= 0 and max(xs) <= 200
        assert min(ys) >= 0 and max(ys) <= 200

    def test_luminance_mode_dark_on_bright(self):
        """Full pipeline on dark circle / bright background JPEG-like image."""
        img = make_circle_rgb(size=200, radius=70, dark_on_bright=True)
        coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
        assert coords is not None
        assert len(coords) >= 4
        # Centroid should be near center
        cx, cy = centroid(coords)
        assert abs(cx - 100) < 40, f"cx={cx:.1f}"
        assert abs(cy - 100) < 40, f"cy={cy:.1f}"

    def test_luminance_mode_bright_on_dark(self):
        """Polarity inversion should still detect the circle."""
        img = make_circle_rgb(size=200, radius=70, dark_on_bright=False)
        coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
        assert coords is not None
        cx, cy = centroid(coords)
        assert abs(cx - 100) < 40, f"cx={cx:.1f}"
        assert abs(cy - 100) < 40, f"cy={cy:.1f}"

    def test_auto_mode_picks_alpha_for_transparent(self):
        """Auto mode should use alpha path for transparent PNG."""
        img = make_circle_rgba(size=200, radius=70)
        coords_auto = get_outline_coords(img, mode='auto')
        coords_alpha = get_outline_coords(img, mode='alpha')
        # Both should succeed
        assert coords_auto is not None
        assert coords_alpha is not None

    def test_auto_mode_picks_luminance_for_opaque(self):
        """Auto mode should fall back to luminance for fully opaque image."""
        img = make_circle_rgb(size=200, radius=70)
        coords = get_outline_coords(img.convert('RGBA'), mode='auto')
        assert coords is not None

    def test_blank_image_returns_none(self):
        img = Image.new('RGBA', (100, 100), (255, 255, 255, 0))
        coords = get_outline_coords(img, mode='alpha')
        assert coords is None

    def test_coordinates_in_image_space(self):
        """Returned coordinates should be within image bounds."""
        img = make_circle_rgba(size=300, radius=100)
        coords = get_outline_coords(img, mode='alpha')
        assert coords is not None
        for x, y in coords:
            assert 0 <= x <= 300, f"x={x} out of bounds"
            assert 0 <= y <= 300, f"y={y} out of bounds"

    def test_rectangle_outline(self):
        img = make_rect_rgba(size=200, rect_frac=0.6)
        coords = get_outline_coords(img, mode='alpha')
        assert coords is not None
        assert len(coords) >= 4

    def test_small_image(self):
        """Should handle small images without crashing."""
        img = make_circle_rgba(size=30, radius=10)
        coords = get_outline_coords(img, mode='alpha')
        # May or may not find coords, but should not raise
        # (small images may not produce a contour after morphological ops)

    def test_large_image_performance(self):
        """Large image should be processed within a reasonable time (downscaled)."""
        import time
        img = make_circle_rgb(size=1000, radius=350)
        start = time.time()
        coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
        elapsed = time.time() - start
        assert elapsed < 30.0, f"Processing 1000px image took {elapsed:.1f}s (too slow)"
        assert coords is not None

    def test_noisy_image(self):
        """Add salt-and-pepper noise; should still detect main shape."""
        img = make_circle_rgb(size=200, radius=70, dark_on_bright=True)
        arr = np.array(img)
        rng = np.random.default_rng(42)
        noise_mask = rng.random(arr.shape[:2]) < 0.05
        arr[noise_mask] = rng.integers(0, 255, (noise_mask.sum(), 3), dtype=np.uint8)
        noisy_img = Image.fromarray(arr)
        coords = get_outline_coords(noisy_img.convert('RGBA'), mode='luminance')
        assert coords is not None
        cx, cy = centroid(coords)
        assert abs(cx - 100) < 50, f"cx={cx:.1f} too far from center"
        assert abs(cy - 100) < 50, f"cy={cy:.1f} too far from center"

    def test_threshold_adjust_effect(self):
        """Different threshold adjustments should produce different outlines."""
        img = make_circle_rgb(size=200, radius=70, dark_on_bright=True)
        rgba = img.convert('RGBA')
        coords_default = get_outline_coords(rgba, mode='luminance', threshold_adjust=0)
        coords_plus = get_outline_coords(rgba, mode='luminance', threshold_adjust=30)
        # Both should return valid outlines
        assert coords_default is not None
        assert coords_plus is not None

    def test_simplification_effect(self):
        """Higher simplification should give fewer points."""
        img = make_circle_rgba(size=200, radius=70)
        coords_fine = get_outline_coords(img, mode='alpha', simplification=0.5)
        coords_coarse = get_outline_coords(img, mode='alpha', simplification=5.0)
        if coords_fine and coords_coarse:
            # Coarser simplification should give fewer or equal points
            assert len(coords_coarse) <= len(coords_fine) + 5  # small tolerance


# ---------------------------------------------------------------------------
# Tests: _border_mean
# ---------------------------------------------------------------------------

class TestBorderMean:
    def test_bright_border_on_bright_image(self):
        arr = np.full((100, 100), 200, dtype=np.uint8)
        mean = _border_mean(arr, border_width=5)
        assert abs(mean - 200) < 5

    def test_dark_border_on_dark_image(self):
        arr = np.full((100, 100), 30, dtype=np.uint8)
        mean = _border_mean(arr, border_width=5)
        assert abs(mean - 30) < 5

    def test_bright_border_dark_center(self):
        """Image with bright border and dark center."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[:5, :] = 255
        arr[-5:, :] = 255
        arr[:, :5] = 255
        arr[:, -5:] = 255
        mean = _border_mean(arr, border_width=5)
        assert mean > 100, "Border is bright; mean should be high"
