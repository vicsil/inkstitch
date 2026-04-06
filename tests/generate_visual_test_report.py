#!/usr/bin/env python3
"""Generate a visual HTML test report for the image_outline pipeline.

Creates synthetic images that simulate real-world conditions,
runs get_outline_coords on each, and produces an HTML report
with SVG overlays for visual inspection.

Usage: python3 tests/generate_visual_test_report.py
Output: tests/image_outline_outputs/report.html
"""

import sys
import os
import io
import math
import base64
import importlib.util
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Direct import (avoids wx dependency)
_module_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'extensions', 'utils', 'image_outline.py')
_spec = importlib.util.spec_from_file_location('image_outline', _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_outline_coords = _mod.get_outline_coords

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'image_outline_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def img_to_b64(img, fmt='PNG'):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def polygon_area(coords):
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


def make_svg_overlay(img, coords, stroke_color='red', stroke_width=2):
    """Return SVG markup embedding the image with the contour overlaid."""
    w, h = img.size
    b64 = img_to_b64(img)
    mime = 'image/png'
    if coords:
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)
        poly = (f'<polyline points="{pts}" '
                f'style="fill:none;stroke:{stroke_color};'
                f'stroke-width:{stroke_width};stroke-opacity:0.9" />')
        # Draw start point marker
        x0, y0 = coords[0]
        marker = (f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{stroke_width*2}" '
                  f'fill="lime" stroke="none" opacity="0.9"/>')
    else:
        poly = '<!-- no contour -->'
        marker = ''

    return f'''<svg xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink"
     width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <image x="0" y="0" width="{w}" height="{h}"
         xlink:href="data:{mime};base64,{b64}" />
  {poly}
  {marker}
</svg>'''


# ---------------------------------------------------------------------------
# Test case generators
# ---------------------------------------------------------------------------

SIZE = 250

def case_circle_transparent():
    img = Image.new('RGBA', (SIZE, SIZE), (240, 240, 240, 0))
    draw = ImageDraw.Draw(img)
    r = 90
    draw.ellipse([SIZE//2-r, SIZE//2-r, SIZE//2+r, SIZE//2+r], fill=(220, 50, 50, 255))
    return img, 'alpha', 'Circle – transparent PNG (alpha mode)'

def case_circle_dark_on_white():
    img = Image.new('RGB', (SIZE, SIZE), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    r = 90
    draw.ellipse([SIZE//2-r, SIZE//2-r, SIZE//2+r, SIZE//2+r], fill=(30, 30, 30))
    return img.convert('RGBA'), 'luminance', 'Circle – dark on white (luminance mode)'

def case_circle_white_on_dark():
    img = Image.new('RGB', (SIZE, SIZE), (25, 25, 25))
    draw = ImageDraw.Draw(img)
    r = 90
    draw.ellipse([SIZE//2-r, SIZE//2-r, SIZE//2+r, SIZE//2+r], fill=(230, 230, 230))
    return img.convert('RGBA'), 'luminance', 'Circle – white on dark (polarity flip)'

def case_star_transparent():
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, ro, ri = SIZE//2, SIZE//2, 100, 42
    pts = []
    for i in range(10):
        angle = math.pi * i / 5 - math.pi / 2
        r = ro if i % 2 == 0 else ri
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, fill=(255, 180, 0, 255))
    return img, 'alpha', 'Star – non-convex shape (alpha mode)'

def case_noisy_photo():
    """Dark subject on white, with salt-and-pepper noise."""
    img = Image.new('RGB', (SIZE, SIZE), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 190, 190], fill=(40, 40, 40))
    # Add S&P noise
    arr = np.array(img)
    rng = np.random.default_rng(7)
    noise = rng.random(arr.shape[:2])
    arr[noise < 0.04] = [255, 255, 255]
    arr[noise > 0.96] = [0, 0, 0]
    return Image.fromarray(arr).convert('RGBA'), 'luminance', 'Noisy photo – S&P noise 4% (luminance mode)'

def case_gradient_bg():
    """Subject on gradient background — hardest case."""
    img = Image.new('RGB', (SIZE, SIZE))
    for x in range(SIZE):
        v = int(200 + x * 50 / SIZE)   # 200→250 left-to-right
        for y in range(SIZE):
            img.putpixel((x, y), (v, v, v))
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 60, 190, 190], fill=(30, 30, 30))
    return img.convert('RGBA'), 'luminance', 'Gradient background – difficult case'

def case_jpeg_like_compression():
    """Simulate JPEG compression artifacts by round-tripping through JPEG."""
    img = Image.new('RGB', (SIZE, SIZE), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.ellipse([55, 55, 195, 195], fill=(25, 35, 80))
    # Simulate JPEG artifacts
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=35)
    buf.seek(0)
    img_jpeg = Image.open(buf).copy()
    return img_jpeg.convert('RGBA'), 'luminance', 'JPEG artefacts – quality=35 (luminance mode)'

def case_two_objects_keep_largest():
    """Two circles: pipeline should keep only the larger one."""
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 180, 180], fill=(100, 150, 220, 255))   # large
    draw.ellipse([210, 210, 240, 240], fill=(220, 80, 80, 255))    # small noise
    return img, 'alpha', 'Two objects – only largest kept (alpha mode)'

def case_thin_rectangle():
    """Very thin horizontal bar."""
    img = Image.new('RGBA', (SIZE, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, SIZE-20, 60], fill=(0, 120, 200, 255))
    return img, 'alpha', 'Thin horizontal bar – extreme aspect ratio'

def case_anti_aliased_circle():
    """Circle with Gaussian-blurred edges (simulates anti-aliasing / soft PNG)."""
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = 90
    draw.ellipse([SIZE//2-r, SIZE//2-r, SIZE//2+r, SIZE//2+r], fill=(180, 60, 200, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    return img, 'alpha', 'Anti-aliased circle – soft edges (alpha mode)'

def case_logo_like_with_hole():
    """Ring/donut shape: exterior circle with transparent interior."""
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 30, 220, 220], fill=(60, 60, 180, 255))
    draw.ellipse([80, 80, 170, 170], fill=(0, 0, 0, 0))   # punch hole
    return img, 'alpha', 'Donut / ring – exterior outline traced (alpha mode)'

def case_auto_transparent():
    """Auto mode on transparent PNG."""
    img = Image.new('RGBA', (SIZE, SIZE), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 200, 200], fill=(80, 200, 100, 255))
    return img, 'auto', 'Auto mode – transparent PNG (should choose alpha)'

def case_auto_opaque():
    """Auto mode on opaque RGB image."""
    img = Image.new('RGB', (SIZE, SIZE), (230, 225, 220))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 200, 200], fill=(20, 20, 60))
    return img.convert('RGBA'), 'auto', 'Auto mode – opaque image (should choose luminance)'

def case_large_1000px():
    """1000px image – tests downscaling performance."""
    BIG = 1000
    img = Image.new('RGBA', (BIG, BIG), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 100, 900, 900], fill=(200, 80, 60, 255))
    return img, 'alpha', 'Large image 1000×1000 – performance test'

ALL_CASES = [
    case_circle_transparent,
    case_circle_dark_on_white,
    case_circle_white_on_dark,
    case_star_transparent,
    case_noisy_photo,
    case_gradient_bg,
    case_jpeg_like_compression,
    case_two_objects_keep_largest,
    case_thin_rectangle,
    case_anti_aliased_circle,
    case_logo_like_with_hole,
    case_auto_transparent,
    case_auto_opaque,
    case_large_1000px,
]


# ---------------------------------------------------------------------------
# Run tests and build report
# ---------------------------------------------------------------------------

def run_all():
    results = []
    for case_fn in ALL_CASES:
        img, mode, desc = case_fn()
        w, h = img.size
        img_area = w * h

        t0 = time.time()
        coords = get_outline_coords(img, mode=mode)
        elapsed = time.time() - t0

        if coords:
            area = polygon_area(coords)
            coverage = area / img_area
            n_pts = len(coords)
            cx, cy = centroid(coords)
            status = 'OK'
            detail = (f'{n_pts} points, area={area:.0f}px² '
                      f'({coverage*100:.1f}%), centroid=({cx:.0f},{cy:.0f})')
        else:
            area = coverage = n_pts = 0
            cx = cy = 0
            status = 'NO CONTOUR'
            detail = 'No contour detected'

        svg = make_svg_overlay(img, coords)
        results.append({
            'desc': desc,
            'mode': mode,
            'size': f'{w}×{h}',
            'status': status,
            'detail': detail,
            'elapsed': elapsed,
            'svg': svg,
            'coords_count': n_pts,
        })
        print(f'  {"✓" if status == "OK" else "✗"} [{elapsed:.2f}s] {desc}: {detail}')

    return results


def build_html_report(results):
    rows = []
    for i, r in enumerate(results):
        color = '#e8f5e9' if r['status'] == 'OK' else '#ffebee'
        rows.append(f'''
    <tr style="background:{color}">
      <td style="padding:8px;font-weight:bold">{r["desc"]}</td>
      <td style="padding:8px;text-align:center"><code>{r["mode"]}</code></td>
      <td style="padding:8px;text-align:center">{r["size"]}</td>
      <td style="padding:8px;text-align:center;font-weight:bold;
                 color:{"green" if r["status"]=="OK" else "red"}">{r["status"]}</td>
      <td style="padding:8px;font-size:0.9em">{r["detail"]}</td>
      <td style="padding:8px;text-align:center">{r["elapsed"]*1000:.0f} ms</td>
    </tr>
    <tr>
      <td colspan="6" style="padding:8px;background:#fafafa">
        <details>
          <summary style="cursor:pointer;color:#555">SVG Overlay (click to expand)</summary>
          <div style="margin-top:8px;overflow:auto">{r["svg"]}</div>
        </details>
      </td>
    </tr>''')

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_total = len(results)

    html = f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Image Outline Pipeline – Visual Test Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #333; color: #fff; padding: 8px; text-align: left; }}
    td {{ border-bottom: 1px solid #ddd; }}
    .summary {{ font-size: 1.2em; margin: 12px 0; }}
  </style>
</head>
<body>
  <h1>Image Outline Pipeline – Visual Test Report</h1>
  <p class="summary">
    <strong>{n_ok}/{n_total}</strong> cases produced a contour.
    Red contour = detected outline. Green dot = start point.
  </p>
  <table>
    <tr>
      <th>Test Case</th><th>Mode</th><th>Size</th>
      <th>Status</th><th>Details</th><th>Time</th>
    </tr>
    {''.join(rows)}
  </table>
</body>
</html>'''
    return html


if __name__ == '__main__':
    print('Running image outline visual tests...\n')
    results = run_all()
    html = build_html_report(results)

    report_path = os.path.join(OUTPUT_DIR, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    n_total = len(results)
    print(f'\n{"="*60}')
    print(f'Results: {n_ok}/{n_total} produced contours')
    print(f'Report : {report_path}')
    if n_ok < n_total:
        sys.exit(1)
