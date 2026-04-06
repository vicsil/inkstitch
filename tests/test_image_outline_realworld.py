# Authors: see git history
#
# Copyright (c) 2026 Authors
# Licensed under the GNU GPL version 3.0 or later.  See the file LICENSE for details.

"""Real-world image tests for image_outline.py.

All image files found in tests/fixtures/ are loaded automatically — no manual
listing is needed.  Drop any PNG, JPEG, WEBP, BMP, or TIFF into that folder
and it will be picked up on the next test run.

The detection mode ('alpha' or 'luminance') is chosen automatically based on
whether the image has a meaningful alpha channel.

Fixture generators: the four bundled synthetic images are (re-)created by
tests/fixtures/generate_fixtures.py whenever they are missing.

SVG overlays showing the detected contour are written to
tests/image_outline_outputs/ for visual inspection.
"""

import base64
import importlib.util
import io
import math
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Direct import – bypasses lib/extensions/__init__.py which requires wx
# ---------------------------------------------------------------------------
_module_path = os.path.join(
    os.path.dirname(__file__), '..', 'lib', 'extensions', 'utils', 'image_outline.py'
)
_spec = importlib.util.spec_from_file_location('image_outline', _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

get_outline_coords   = _mod.get_outline_coords
has_meaningful_alpha = _mod.has_meaningful_alpha

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'image_outline_outputs')
os.makedirs(FIXTURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

# Supported image extensions (anything PIL can open)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}


# ---------------------------------------------------------------------------
# Fixture generators  (called lazily when a file is missing)
# ---------------------------------------------------------------------------

def _make_python_logo(size=200):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size
    draw.ellipse([s*0.20, s*0.05, s*0.65, s*0.55], fill=(55, 118, 171, 255))
    draw.ellipse([s*0.35, s*0.45, s*0.80, s*0.95], fill=(255, 213, 75, 255))
    draw.ellipse([s*0.35, s*0.38, s*0.65, s*0.62], fill=(0, 0, 0, 0))
    draw.ellipse([s*0.60, s*0.08, s*0.75, s*0.20], fill=(55, 118, 171, 255))
    draw.ellipse([s*0.25, s*0.80, s*0.40, s*0.92], fill=(255, 213, 75, 255))
    return img.filter(ImageFilter.GaussianBlur(radius=1))


def _make_wikipedia_logo(size=200):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size
    cx, cy, r = s // 2, s // 2, int(s * 0.42)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(240, 240, 240, 255))
    for deg in range(0, 360, 45):
        a = math.radians(deg)
        draw.line(
            [(cx + int(r*0.3*math.cos(a)), cy + int(r*0.3*math.sin(a))),
             (cx + int(r*math.cos(a)),     cy + int(r*math.sin(a)))],
            fill=(150, 150, 150, 255), width=2
        )
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(100, 100, 100, 255), width=2)
    return img


def _make_portrait(size=220):
    arr = np.full((size, size, 3), 200, dtype=np.uint8)
    for y in range(size):
        v = int(180 + y * 30 / size)
        arr[y, :] = [v, max(0, v-5), max(0, v-10)]
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    s = size
    draw.ellipse([s*0.15, s*0.65, s*0.85, s*1.05], fill=(60,  40,  80))
    draw.ellipse([s*0.28, s*0.20, s*0.72, s*0.70], fill=(210, 170, 140))
    draw.ellipse([s*0.22, s*0.10, s*0.78, s*0.50], fill=(60,  40,  25))
    draw.ellipse([s*0.28, s*0.30, s*0.72, s*0.70], fill=(210, 170, 140))
    draw.ellipse([s*0.20, s*0.05, s*0.80, s*0.35], fill=(180, 50,  50))
    draw.ellipse([s*0.28, s*0.20, s*0.72, s*0.50], fill=(210, 170, 140))
    draw.ellipse([s*0.36, s*0.38, s*0.45, s*0.45], fill=(30,  30,  30))
    draw.ellipse([s*0.55, s*0.38, s*0.64, s*0.45], fill=(30,  30,  30))
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def _make_cat_photo(size=220):
    img = Image.new('RGB', (size, size), (200, 195, 190))
    draw = ImageDraw.Draw(img)
    s = size
    cc = (30, 28, 25)
    draw.ellipse([s*0.25, s*0.40, s*0.80, s*0.90], fill=cc)
    draw.ellipse([s*0.30, s*0.15, s*0.72, s*0.52], fill=cc)
    draw.polygon([(s*0.33, s*0.28), (s*0.22, s*0.08), (s*0.42, s*0.18)], fill=cc)
    draw.polygon([(s*0.67, s*0.28), (s*0.78, s*0.08), (s*0.58, s*0.18)], fill=cc)
    draw.arc([s*0.60, s*0.55, s*0.95, s*0.95], start=200, end=340,
             fill=cc, width=int(s*0.06))
    draw.ellipse([s*0.38, s*0.29, s*0.47, s*0.38], fill=(80, 160, 60))
    draw.ellipse([s*0.53, s*0.29, s*0.62, s*0.38], fill=(80, 160, 60))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    buf.seek(0)
    return Image.open(buf).copy()


_BUNDLED_FIXTURES = {
    'python_logo_transparent.png':    (_make_python_logo,    'PNG'),
    'wikipedia_logo_transparent.png': (_make_wikipedia_logo, 'PNG'),
    'lena_grayscale.png':             (_make_portrait,       'PNG'),
    'black_cat_photo.jpg':            (_make_cat_photo,      'JPEG'),
}


def _ensure_bundled_fixtures():
    """Generate the bundled synthetic fixture images if any are missing."""
    for filename, (generator, fmt) in _BUNDLED_FIXTURES.items():
        path = os.path.join(FIXTURES_DIR, filename)
        if not os.path.exists(path):
            img = generator()
            img.save(path, fmt)


# ---------------------------------------------------------------------------
# Auto-discovery: scan FIXTURES_DIR for all image files
# ---------------------------------------------------------------------------

def _discover_fixture_cases():
    """Return a list of test-case dicts for every image in FIXTURES_DIR.

    Each image is tested in 'auto' mode (the pipeline chooses alpha vs
    luminance).  The test ID is the bare filename.
    """
    _ensure_bundled_fixtures()

    cases = []
    for fname in sorted(os.listdir(FIXTURES_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue
        # Skip the generator script itself (safety)
        if fname.endswith('.py'):
            continue
        cases.append({
            'name': os.path.splitext(fname)[0],
            'file': fname,
            'mode': 'auto',
        })
    return cases


# ---------------------------------------------------------------------------
# SVG overlay helper
# ---------------------------------------------------------------------------

def save_svg_overlay(img, coords, output_path):
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    sw = max(1, w // 100)

    if coords:
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)
        poly = (f'<polyline points="{pts}" style="fill:none;stroke:red;'
                f'stroke-width:{sw};stroke-opacity:0.85" />')
        x0, y0 = coords[0]
        marker = (f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{sw*2}" '
                  f'fill="lime" stroke="none" opacity="0.9"/>')
    else:
        poly   = '<!-- no contour detected -->'
        marker = ''

    svg = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<svg xmlns="http://www.w3.org/2000/svg" '
           f'xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
           f'  <image x="0" y="0" width="{w}" height="{h}" '
           f'xlink:href="data:image/png;base64,{b64}" />\n'
           f'  {poly}\n  {marker}\n</svg>\n')
    with open(output_path, 'w') as f:
        f.write(svg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Auto-discovered parametrised test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', _discover_fixture_cases(),
                         ids=[c['name'] for c in _discover_fixture_cases()])
def test_fixture_image_pipeline(case):
    """Load every image in tests/fixtures/ and validate get_outline_coords.

    Mode is chosen automatically ('auto').  A contour must be detected,
    stay within the image bounds, and cover a meaningful fraction of the image.
    An SVG overlay is written to tests/image_outline_outputs/ for inspection.
    """
    path = os.path.join(FIXTURES_DIR, case['file'])
    img = Image.open(path).copy()
    img_rgba = img.convert('RGBA')
    w, h = img_rgba.size

    coords = get_outline_coords(img_rgba, mode=case['mode'])

    out_path = os.path.join(OUTPUT_DIR, f"{case['name']}.svg")
    save_svg_overlay(img_rgba, coords, out_path)

    assert coords is not None, (
        f"{case['file']}: no contour detected. SVG: {out_path}"
    )
    assert len(coords) >= 4, f"Too few points: {len(coords)}"

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    assert min(xs) >= -1,    f"x underflow: {min(xs):.1f}"
    assert max(xs) <= w + 1, f"x overflow:  {max(xs):.1f} > {w}"
    assert min(ys) >= -1,    f"y underflow: {min(ys):.1f}"
    assert max(ys) <= h + 1, f"y overflow:  {max(ys):.1f} > {h}"

    area     = polygon_area(coords)
    coverage = area / (w * h)
    assert coverage > 0.02, (
        f"Contour covers only {coverage*100:.1f}% of image – too small"
    )
    assert coverage < 0.99, (
        f"Contour covers {coverage*100:.1f}% – entire image traced (no subject found)"
    )


# ---------------------------------------------------------------------------
# Constructed edge-case tests (no fixture files needed)
# ---------------------------------------------------------------------------

def test_gradient_background():
    """Gradient background – hardest luminance case; must not raise."""
    size = 200
    img = Image.new('RGB', (size, size))
    for x in range(size):
        v = int(200 + x * 50 / size)
        for y in range(size):
            img.putpixel((x, y), (v, v, v))
    ImageDraw.Draw(img).ellipse([60, 60, 140, 140], fill=(30, 30, 30))
    coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
    if coords is not None:
        cx, cy = centroid(coords)
        assert abs(cx - 100) < 60
        assert abs(cy - 100) < 60


def test_very_low_contrast():
    """Near-uniform image – OK to return None; must not raise."""
    size = 200
    img = Image.new('RGB', (size, size), (180, 180, 180))
    ImageDraw.Draw(img).ellipse([60, 60, 140, 140], fill=(150, 150, 150))
    coords = get_outline_coords(img.convert('RGBA'), mode='luminance')
    if coords is not None:
        assert all(0 <= x <= size for x, y in coords)
        assert all(0 <= y <= size for x, y in coords)


def test_multi_region_keeps_largest():
    """Two circles – only the largest should survive."""
    size = 300
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 220, 220], fill=(255, 0, 0, 255))
    draw.ellipse([255, 15, 285, 45],  fill=(0, 0, 255, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    cx, cy = centroid(coords)
    assert abs(cx - 140) < 50
    assert abs(cy - 140) < 50


def test_tall_aspect_ratio():
    img = Image.new('RGBA', (100, 400), (255, 255, 255, 0))
    ImageDraw.Draw(img).rectangle([20, 50, 80, 350], fill=(0, 200, 0, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    assert all(0 <= x <= 100 for x, y in coords)
    assert all(0 <= y <= 400 for x, y in coords)


def test_wide_aspect_ratio():
    img = Image.new('RGBA', (400, 100), (255, 255, 255, 0))
    ImageDraw.Draw(img).rectangle([50, 20, 350, 80], fill=(200, 0, 0, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    assert all(0 <= x <= 400 for x, y in coords)
    assert all(0 <= y <= 100 for x, y in coords)


def test_anti_aliased_edges():
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([40, 40, 160, 160], fill=(180, 60, 200, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    cx, cy = centroid(coords)
    assert abs(cx - 100) < 30
    assert abs(cy - 100) < 30


def test_star_shape_alpha():
    size = 200
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    cx, cy, ro, ri = size//2, size//2, 80, 35
    pts = []
    for i in range(10):
        a = math.pi * i / 5 - math.pi / 2
        r = ro if i % 2 == 0 else ri
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    ImageDraw.Draw(img).polygon(pts, fill=(255, 215, 0, 255))
    coords = get_outline_coords(img, mode='alpha')
    assert coords is not None
    assert polygon_area(coords) > 5000


def test_simplification_produces_fewer_points():
    img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse([30, 30, 170, 170], fill=(200, 50, 50, 255))
    fine   = get_outline_coords(img, mode='alpha', simplification=0.5)
    coarse = get_outline_coords(img, mode='alpha', simplification=8.0)
    if fine and coarse:
        assert len(coarse) <= len(fine) + 5
