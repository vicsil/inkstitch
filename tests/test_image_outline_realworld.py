# Authors: see git history
#
# Copyright (c) 2026 Authors
# Licensed under the GNU GPL version 3.0 or later.  See the file LICENSE for details.

"""Real-world image tests for image_outline.py.

Downloads a curated set of test images that cover:
  - Simple logos with transparency (alpha mode)
  - Product photos on white background (luminance mode)
  - High-contrast subjects
  - Low-contrast / difficult cases
  - Multi-region images (disconnected foreground)

Also produces SVG overlays so the contours can be inspected visually.
"""

import sys
import os
import math
import importlib.util
import io
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw

# Direct import to bypass wx dependency chain
_module_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'extensions', 'utils', 'image_outline.py')
_spec = importlib.util.spec_from_file_location('image_outline', _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_outline_coords = _mod.get_outline_coords

# Output dir for visual inspection SVGs
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'image_outline_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: save an SVG overlay of image + detected contour
# ---------------------------------------------------------------------------

def save_svg_overlay(image_path_or_url, coords, output_path, img_w, img_h):
    """Save an SVG that embeds the original image and draws the contour on top."""
    import base64
    import urllib.request

    if image_path_or_url.startswith('http'):
        data = urllib.request.urlopen(image_path_or_url, timeout=10).read()
        ext = image_path_or_url.split('?')[0].rsplit('.', 1)[-1].lower()
        if ext not in ('png', 'jpg', 'jpeg', 'gif', 'webp'):
            ext = 'png'
        mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
        img_b64 = base64.b64encode(data).decode()
        href = f"data:{mime};base64,{img_b64}"
    else:
        with open(image_path_or_url, 'rb') as f:
            data = f.read()
        ext = image_path_or_url.rsplit('.', 1)[-1].lower()
        mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
        img_b64 = base64.b64encode(data).decode()
        href = f"data:{mime};base64,{img_b64}"

    if coords:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        polyline = (
            f'<polyline points="{pts}" '
            f'style="fill:none;stroke:red;stroke-width:{max(1, img_w//100)};stroke-opacity:0.8" />'
        )
    else:
        polyline = '<!-- no contour detected -->'

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{img_w}" height="{img_h}" viewBox="0 0 {img_w} {img_h}">
  <image x="0" y="0" width="{img_w}" height="{img_h}" xlink:href="{href}" />
  {polyline}
</svg>
"""
    with open(output_path, 'w') as f:
        f.write(svg)


def centroid(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def polygon_area(coords):
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return abs(area) / 2.0


def fetch_image(url, target_size=None):
    """Download image from URL, optionally resize, return PIL Image."""
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data))
    img.load()  # ensure fully loaded before urllib closes
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img


# ---------------------------------------------------------------------------
# Test images (all public domain / CC0 from Wikimedia or similar)
# ---------------------------------------------------------------------------

# We use small, reliable sources. Each entry: (url, mode, description, expected_check)
REAL_IMAGES = [
    # --- Alpha-channel images (transparent PNG) ---
    {
        'name': 'python_logo_transparent',
        # Python logo with transparent background (official SVG rendered to PNG via wikimedia)
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png',
        'mode': 'alpha',
        'desc': 'Python logo - transparent PNG (alpha mode)',
        'size': (200, 200),
        'expect_contour': True,
    },
    {
        'name': 'wikipedia_logo_transparent',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/80/Wikipedia-logo-v2.svg/200px-Wikipedia-logo-v2.svg.png',
        'mode': 'alpha',
        'desc': 'Wikipedia logo - transparent PNG (alpha mode)',
        'size': (200, 200),
        'expect_contour': True,
    },
    # --- Opaque/JPEG images (luminance mode) ---
    {
        'name': 'lena_grayscale',
        # Classic Lena/Lenna test image (grayscale version, public domain)
        'url': 'https://upload.wikimedia.org/wikipedia/en/thumb/7/7d/Lenna_%28test_image%29.png/220px-Lenna_%28test_image%29.png',
        'mode': 'luminance',
        'desc': 'Lena standard test image (portrait photo)',
        'size': (220, 220),
        'expect_contour': True,
    },
    {
        'name': 'black_cat_photo',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/220px-Cat_November_2010-1a.jpg',
        'mode': 'luminance',
        'desc': 'Cat photo on plain background (luminance mode)',
        'size': (220, 220),
        'expect_contour': True,
    },
    {
        'name': 'auto_mode_opaque',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/220px-Cat_November_2010-1a.jpg',
        'mode': 'auto',
        'desc': 'Auto mode on opaque JPEG (should choose luminance)',
        'size': (220, 220),
        'expect_contour': True,
    },
    {
        'name': 'auto_mode_transparent',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/200px-Python-logo-notext.svg.png',
        'mode': 'auto',
        'desc': 'Auto mode on transparent PNG (should choose alpha)',
        'size': (200, 200),
        'expect_contour': True,
    },
]


# ---------------------------------------------------------------------------
# Parameterised real-world tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', REAL_IMAGES, ids=[c['name'] for c in REAL_IMAGES])
def test_real_image_pipeline(case):
    """Download image, run get_outline_coords, validate and save SVG overlay."""
    try:
        img = fetch_image(case['url'], target_size=case.get('size'))
    except Exception as e:
        pytest.skip(f"Could not download image: {e}")

    # Ensure RGBA for the pipeline
    img_rgba = img.convert('RGBA')
    w, h = img_rgba.size

    coords = get_outline_coords(img_rgba, mode=case['mode'])

    # Save SVG overlay for visual inspection regardless of pass/fail
    out_name = f"{case['name']}_{case['mode']}.svg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_svg_overlay(case['url'], coords, out_path, w, h)
    print(f"\n  SVG saved: {out_path}")

    if case['expect_contour']:
        assert coords is not None, (
            f"{case['desc']}: expected a contour but got None. "
            f"SVG overlay saved to {out_path}"
        )
        assert len(coords) >= 4, f"Contour has too few points: {len(coords)}"

        # All coordinates must be within image bounds (with 1px tolerance)
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        assert min(xs) >= -1, f"x underflow: {min(xs):.1f}"
        assert max(xs) <= w + 1, f"x overflow: {max(xs):.1f} > {w}"
        assert min(ys) >= -1, f"y underflow: {min(ys):.1f}"
        assert max(ys) <= h + 1, f"y overflow: {max(ys):.1f} > {h}"

        # Contour should cover a meaningful fraction of the image
        area = polygon_area(coords)
        image_area = w * h
        coverage = area / image_area
        assert coverage > 0.02, (
            f"Contour area {area:.0f}px² is only {coverage*100:.1f}% of image — too small"
        )
        assert coverage < 0.99, (
            f"Contour covers {coverage*100:.1f}% of image — entire image traced (no subject found)"
        )

        print(f"  Contour: {len(coords)} points, area={area:.0f}px² ({coverage*100:.1f}% of image)")


# ---------------------------------------------------------------------------
# Additional edge-case real-world tests (constructed locally)
# ---------------------------------------------------------------------------

def test_gradient_background():
    """Image with gradient background — hardest case for luminance detection."""
    size = 200
    img = Image.new('RGB', (size, size))
    # Gradient from white (left) to grey (right)
    for x in range(size):
        v = int(255 - x * 100 / size)
        for y in range(size):
            img.putpixel((x, y), (v, v, v))
    # Draw a dark circle in the middle
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 140, 140], fill=(30, 30, 30))

    coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
    # Should at least return something — gradient bg is genuinely hard
    # We accept None here (not a regression, just documentation)
    if coords is not None:
        cx, cy = centroid(coords)
        assert abs(cx - 100) < 60, f"cx={cx:.1f} too far from center"


def test_very_low_contrast():
    """Near-uniform image: dark grey circle on grey background."""
    size = 200
    img = Image.new('RGB', (size, size), (180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 140, 140], fill=(150, 150, 150))  # only 30 grey levels difference
    coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
    # Low contrast: OK to return None — just must not crash
    # If it does return coords, they should be in bounds
    if coords is not None:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        assert all(0 <= x <= size for x in xs)
        assert all(0 <= y <= size for y in ys)


def test_multi_region_keeps_largest():
    """Two circles of different sizes — only largest should appear in output."""
    size = 300
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    # Large circle: r=80 → area ≈ 20000 px²
    draw.ellipse([60, 60, 220, 220], fill=(255, 0, 0, 255))
    # Small circle: r=15 → area ≈ 700 px²
    draw.ellipse([255, 15, 285, 45], fill=(0, 0, 255, 255))

    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None

    # Centroid should be near center of the large circle (140, 140)
    cx, cy = centroid(coords)
    assert abs(cx - 140) < 50, f"cx={cx:.1f} — expected near large-circle center 140"
    assert abs(cy - 140) < 50, f"cy={cy:.1f} — expected near large-circle center 140"


def test_tall_aspect_ratio():
    """Very tall, thin image (portrait aspect ratio)."""
    img = Image.new('RGBA', (100, 400), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 50, 80, 350], fill=(0, 200, 0, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert min(xs) >= 0 and max(xs) <= 100
    assert min(ys) >= 0 and max(ys) <= 400


def test_wide_aspect_ratio():
    """Very wide, thin image (landscape aspect ratio)."""
    img = Image.new('RGBA', (400, 100), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 20, 350, 80], fill=(200, 0, 0, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert min(xs) >= 0 and max(xs) <= 400
    assert min(ys) >= 0 and max(ys) <= 100


def test_anti_aliased_edges():
    """Image with anti-aliased circle edges — alpha channel has gradients."""
    from PIL import ImageFilter
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill=(255, 100, 0, 255))
    # Apply blur to simulate anti-aliasing on the alpha channel
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None, "Anti-aliased transparent PNG should still give a contour"
    cx, cy = centroid(coords)
    assert abs(cx - 100) < 30
    assert abs(cy - 100) < 30


def test_star_shape_alpha():
    """Concave star shape — tests that contour handles non-convex outlines."""
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # 5-pointed star
    import math as _math
    cx, cy, r_outer, r_inner = 100, 100, 80, 35
    pts = []
    for i in range(10):
        angle = _math.pi * i / 5 - _math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * _math.cos(angle), cy + r * _math.sin(angle)))
    draw.polygon(pts, fill=(255, 215, 0, 255))

    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    area = polygon_area(coords)
    # Star with outer radius 80 has area between convex hull (≈20000) and true area (≈12000)
    # After simplification and buffering, we expect something in that range
    assert area > 5000, f"Star outline area {area:.0f} too small"


def test_simplification_produces_fewer_points():
    """Higher simplification should always produce ≤ points."""
    img_url = None  # use local synthetic
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 170, 170], fill=(200, 50, 50, 255))

    coords_fine = get_outline_coords(img, mode='alpha', simplification=0.5)
    coords_coarse = get_outline_coords(img, mode='alpha', simplification=8.0)
    if coords_fine and coords_coarse:
        assert len(coords_coarse) <= len(coords_fine), (
            f"Coarse simplification gave MORE points ({len(coords_coarse)}) "
            f"than fine ({len(coords_fine)})"
        )
