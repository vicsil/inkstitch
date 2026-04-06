#!/usr/bin/env python3
"""Generate a visual HTML test report for the image_outline pipeline.

Scans a folder for image files, runs get_outline_coords on each, and produces
an HTML report with inline SVG overlays for visual inspection.

Usage:
    python3 tests/generate_visual_test_report.py [--input-dir PATH]

  --input-dir  Folder to scan for images (default: tests/fixtures/).
               Any PNG, JPEG, BMP, TIFF, or WEBP file is tested automatically.

Output: tests/image_outline_outputs/report.html
"""

import argparse
import base64
import importlib.util
import io
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Direct import (avoids wx dependency)
_module_path = os.path.join(
    os.path.dirname(__file__), '..', 'lib', 'extensions', 'utils', 'image_outline.py'
)
_spec = importlib.util.spec_from_file_location('image_outline', _module_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_outline_coords   = _mod.get_outline_coords
has_meaningful_alpha = _mod.has_meaningful_alpha

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'image_outline_outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

DEFAULT_INPUT_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


# ---------------------------------------------------------------------------
# Discover all images in a folder
# ---------------------------------------------------------------------------

def discover_images(folder):
    """Return sorted list of (filepath, filename) for every image in *folder*."""
    entries = []
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
            entries.append((os.path.join(folder, fname), fname))
    return entries


# ---------------------------------------------------------------------------
# SVG overlay builder
# ---------------------------------------------------------------------------

def img_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return base64.b64encode(buf.getvalue()).decode()


def make_svg_overlay(img, coords):
    w, h = img.size
    b64 = img_to_b64(img)
    sw = max(1, w // 100)

    if coords:
        pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)
        poly = (f'<polyline points="{pts}" '
                f'style="fill:none;stroke:red;stroke-width:{sw};stroke-opacity:0.9" />')
        x0, y0 = coords[0]
        marker = (f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="{sw*2}" '
                  f'fill="lime" stroke="none" opacity="0.9"/>')
    else:
        poly   = '<!-- no contour detected -->'
        marker = ''

    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<image x="0" y="0" width="{w}" height="{h}" '
            f'xlink:href="data:image/png;base64,{b64}"/>'
            f'{poly}{marker}</svg>')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Run the pipeline on all discovered images
# ---------------------------------------------------------------------------

def run_all(input_dir):
    images = discover_images(input_dir)
    if not images:
        print(f'No image files found in {input_dir}')
        return []

    results = []
    for filepath, fname in images:
        img = Image.open(filepath).convert('RGBA')
        w, h = img.size

        # Auto-detect mode
        mode = 'alpha' if has_meaningful_alpha(img) else 'luminance'

        t0 = time.time()
        coords = get_outline_coords(img, mode='auto')
        elapsed = time.time() - t0

        if coords:
            area     = polygon_area(coords)
            coverage = area / (w * h)
            n_pts    = len(coords)
            cx, cy   = centroid(coords)
            status   = 'OK'
            detail   = (f'{n_pts} pts, area={area:.0f}px² '
                        f'({coverage*100:.1f}%), centroid=({cx:.0f},{cy:.0f})')
        else:
            area = coverage = n_pts = 0
            cx = cy = 0
            status = 'NO CONTOUR'
            detail = 'No contour detected'

        tick = '✓' if status == 'OK' else '✗'
        print(f'  {tick} [{elapsed:.2f}s]  {fname}  ({mode} auto-detected): {detail}')

        results.append({
            'desc':    fname,
            'mode':    f'{mode} (auto)',
            'size':    f'{w}×{h}',
            'status':  status,
            'detail':  detail,
            'elapsed': elapsed,
            'svg':     make_svg_overlay(img, coords),
        })

    return results


# ---------------------------------------------------------------------------
# HTML report builder
# ---------------------------------------------------------------------------

def build_html_report(results, input_dir):
    rows = []
    for r in results:
        bg = '#e8f5e9' if r['status'] == 'OK' else '#ffebee'
        color = 'green' if r['status'] == 'OK' else 'red'
        rows.append(f'''
    <tr style="background:{bg}">
      <td style="padding:8px;font-weight:bold">{r["desc"]}</td>
      <td style="padding:8px;text-align:center"><code>{r["mode"]}</code></td>
      <td style="padding:8px;text-align:center">{r["size"]}</td>
      <td style="padding:8px;text-align:center;font-weight:bold;color:{color}">{r["status"]}</td>
      <td style="padding:8px;font-size:.9em">{r["detail"]}</td>
      <td style="padding:8px;text-align:center">{r["elapsed"]*1000:.0f} ms</td>
    </tr>
    <tr>
      <td colspan="6" style="padding:8px;background:#fafafa">
        <details>
          <summary style="cursor:pointer;color:#555">SVG Overlay</summary>
          <div style="margin-top:8px;overflow:auto">{r["svg"]}</div>
        </details>
      </td>
    </tr>''')

    n_ok    = sum(1 for r in results if r['status'] == 'OK')
    n_total = len(results)

    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Image Outline Pipeline – Visual Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    h1 {{ color: #333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th {{ background: #333; color: #fff; padding: 8px; text-align: left; }}
    td {{ border-bottom: 1px solid #ddd; }}
    .summary {{ font-size: 1.1em; margin: 10px 0 18px; }}
  </style>
</head>
<body>
  <h1>Image Outline Pipeline – Visual Report</h1>
  <p class="summary">
    Input folder: <code>{input_dir}</code><br>
    <strong>{n_ok}/{n_total}</strong> images produced a contour.
    Red line = detected outline &nbsp;·&nbsp; Green dot = path start.
  </p>
  <table>
    <tr>
      <th>File</th><th>Mode</th><th>Size</th>
      <th>Status</th><th>Details</th><th>Time</th>
    </tr>
    {''.join(rows)}
  </table>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir', default=DEFAULT_INPUT_DIR,
                        help='Folder to scan for images (default: tests/fixtures/)')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    print(f'Scanning: {input_dir}\n')

    results = run_all(input_dir)
    if not results:
        sys.exit(1)

    html = build_html_report(results, input_dir)
    report_path = os.path.join(OUTPUT_DIR, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    n_ok = sum(1 for r in results if r['status'] == 'OK')
    print(f'\n{"="*60}')
    print(f'Results : {n_ok}/{len(results)} produced contours')
    print(f'Report  : {report_path}')
    if n_ok < len(results):
        sys.exit(1)
