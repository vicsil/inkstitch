#!/usr/bin/env python3
"""Generate fixture images for test_image_outline_realworld.py.

Creates synthetic images that match the properties of the real-world images
the tests originally tried to download, so tests can run without network access.

Run once: python3 tests/fixtures/generate_fixtures.py
"""

import os
import io
import math
from PIL import Image, ImageDraw, ImageFilter

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def save(img, name, fmt='PNG', quality=None):
    path = os.path.join(FIXTURES_DIR, name)
    kwargs = {}
    if quality is not None:
        kwargs['quality'] = quality
    img.save(path, fmt, **kwargs)
    print(f'  written: {name}  ({img.size[0]}x{img.size[1]}, {img.mode})')
    return path


# ---------------------------------------------------------------------------
# python_logo_transparent.png
# Simulates: Python logo silhouette – two interlocked snake/teardrop shapes
# on a transparent background, in blue/yellow colourway.
# ---------------------------------------------------------------------------
def make_python_logo(size=200):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size

    # Top snake body (blue, offset up-left)
    draw.ellipse([s*0.20, s*0.05, s*0.65, s*0.55], fill=(55, 118, 171, 255))
    draw.ellipse([s*0.35, s*0.45, s*0.80, s*0.95], fill=(255, 213, 75, 255))

    # Overlap mask to create interlock
    draw.ellipse([s*0.35, s*0.38, s*0.65, s*0.62], fill=(0, 0, 0, 0))

    # Add small head/tail dots
    draw.ellipse([s*0.60, s*0.08, s*0.75, s*0.20], fill=(55, 118, 171, 255))
    draw.ellipse([s*0.25, s*0.80, s*0.40, s*0.92], fill=(255, 213, 75, 255))

    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img


# ---------------------------------------------------------------------------
# wikipedia_logo_transparent.png
# Simulates: Wikipedia globe – sphere with text fragments, transparent bg.
# ---------------------------------------------------------------------------
def make_wikipedia_logo(size=200):
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    s = size

    # Main sphere (white with grey shading)
    cx, cy, r = s//2, s//2, int(s*0.42)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(240, 240, 240, 255))

    # Puzzle piece lines
    for angle_deg in range(0, 360, 45):
        a = math.radians(angle_deg)
        x1 = cx + int(r * 0.3 * math.cos(a))
        y1 = cy + int(r * 0.3 * math.sin(a))
        x2 = cx + int(r * math.cos(a))
        y2 = cy + int(r * math.sin(a))
        draw.line([(x1, y1), (x2, y2)], fill=(150, 150, 150, 255), width=2)

    # Outer border
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=(100, 100, 100, 255), width=2)

    return img


# ---------------------------------------------------------------------------
# lena_grayscale.png
# Simulates: Portrait photo – face-like tonal structure, full opaque RGB.
# Lena is a 512×512 classic; we make a 220×220 synthetic portrait.
# ---------------------------------------------------------------------------
def make_portrait(size=220):
    import numpy as np
    rng = np.random.default_rng(42)
    arr = np.full((size, size, 3), 200, dtype=np.uint8)

    # Gradient background (light grey-beige)
    for y in range(size):
        v = int(180 + y * 30 / size)
        arr[y, :] = [v, v-5, v-10]

    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Shoulders/body (dark clothing)
    draw.ellipse([size*0.15, size*0.65, size*0.85, size*1.1],
                 fill=(60, 40, 80))

    # Face (skin tone ellipse)
    draw.ellipse([size*0.28, size*0.20, size*0.72, size*0.70],
                 fill=(210, 170, 140))

    # Hair (dark, top of head)
    draw.ellipse([size*0.22, size*0.10, size*0.78, size*0.50],
                 fill=(60, 40, 25))
    # Erase lower half of hair to show face
    draw.ellipse([size*0.28, size*0.30, size*0.72, size*0.70],
                 fill=(210, 170, 140))

    # Hat / accent
    draw.ellipse([size*0.20, size*0.05, size*0.80, size*0.35],
                 fill=(180, 50, 50))
    draw.ellipse([size*0.28, size*0.20, size*0.72, size*0.50],
                 fill=(210, 170, 140))

    # Eyes
    draw.ellipse([size*0.36, size*0.38, size*0.45, size*0.45], fill=(30, 30, 30))
    draw.ellipse([size*0.55, size*0.38, size*0.64, size*0.45], fill=(30, 30, 30))

    # Light blur for photo-like quality
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img


# ---------------------------------------------------------------------------
# black_cat_photo.jpg
# Simulates: Dark cat on light grey background – classic animal silhouette.
# ---------------------------------------------------------------------------
def make_cat_photo(size=220):
    img = Image.new('RGB', (size, size), (200, 195, 190))
    draw = ImageDraw.Draw(img)
    s = size
    cat_color = (30, 28, 25)

    # Body (large oval)
    draw.ellipse([s*0.25, s*0.40, s*0.80, s*0.90], fill=cat_color)

    # Head (circle)
    draw.ellipse([s*0.30, s*0.15, s*0.72, s*0.52], fill=cat_color)

    # Ears (triangles)
    def triangle(draw, pts, fill):
        draw.polygon(pts, fill=fill)

    triangle(draw, [
        (s*0.33, s*0.28), (s*0.22, s*0.08), (s*0.42, s*0.18)
    ], cat_color)
    triangle(draw, [
        (s*0.67, s*0.28), (s*0.78, s*0.08), (s*0.58, s*0.18)
    ], cat_color)

    # Tail (curved arc – approximate with ellipse segment)
    draw.arc([s*0.60, s*0.55, s*0.95, s*0.95], start=200, end=340,
             fill=cat_color, width=int(s*0.06))

    # Eyes (bright)
    draw.ellipse([s*0.38, s*0.29, s*0.47, s*0.38], fill=(80, 160, 60))
    draw.ellipse([s*0.53, s*0.29, s*0.62, s*0.38], fill=(80, 160, 60))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))

    # Save as JPEG to simulate compression
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# dolphin_embroidery.jpg
# Simulates: blue dolphin embroidery stitched on cream/beige fabric.
# Subject is a coloured blob on a lightly-textured light background.
# ---------------------------------------------------------------------------
def make_dolphin_embroidery(size=240):
    import numpy as np
    rng = np.random.default_rng(7)
    noise = rng.integers(-8, 9, (size, size, 3), dtype=np.int16)
    base = np.array([212, 205, 195], dtype=np.int16)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, 'RGB')
    draw = ImageDraw.Draw(img)
    s = size

    # Dolphin body – horizontal blue oval
    draw.ellipse([s*0.18, s*0.32, s*0.88, s*0.72], fill=(60, 100, 195))
    # Tail fin
    draw.polygon([
        (s*0.82, s*0.42), (s*0.95, s*0.30), (s*0.95, s*0.64)
    ], fill=(50, 85, 175))
    # Dorsal fin
    draw.polygon([
        (s*0.45, s*0.32), (s*0.52, s*0.12), (s*0.60, s*0.32)
    ], fill=(50, 85, 175))
    # Belly (peach)
    draw.ellipse([s*0.22, s*0.52, s*0.72, s*0.70], fill=(200, 165, 140))
    # Eye
    draw.ellipse([s*0.24, s*0.44, s*0.30, s*0.52], fill=(10, 10, 10))

    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=88)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# orange_cat_photo.jpg
# Simulates: orange-and-white cat on complex background (wood planks +
# concrete + grass).  Hardest case: cat has mixed luminance; background
# is also varied.
# ---------------------------------------------------------------------------
def make_orange_cat_photo(size=240):
    import numpy as np
    rng = np.random.default_rng(13)
    arr = np.zeros((size, size, 3), dtype=np.int16)
    s = size

    # Background: wood planks top-left, grass top-right, concrete bottom
    for y in range(size):
        for x in range(size):
            if y < size // 2:
                if x < size // 2:
                    band = (y // 14) % 2
                    v = 130 + band * 18 + int(rng.integers(-6, 7))
                    arr[y, x] = [v - 8, v - 12, v - 18]
                else:
                    v = 115 + int(rng.integers(-12, 13))
                    arr[y, x] = [v - 15, v + 8, v - 18]
            else:
                v = 158 + int(rng.integers(-8, 9))
                arr[y, x] = [v, v, v - 4]

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(img)

    # Cat body – orange
    draw.ellipse([s*0.28, s*0.32, s*0.80, s*0.84], fill=(205, 118, 45))
    # Head
    draw.ellipse([s*0.30, s*0.10, s*0.72, s*0.46], fill=(200, 115, 42))
    # White belly
    draw.ellipse([s*0.36, s*0.54, s*0.68, s*0.80], fill=(238, 232, 220))
    # White face patch
    draw.ellipse([s*0.36, s*0.22, s*0.64, s*0.44], fill=(235, 228, 215))
    # Orange patches back over white face
    draw.ellipse([s*0.36, s*0.20, s*0.54, s*0.30], fill=(200, 115, 42))
    # Ears
    draw.polygon([(s*0.33, s*0.20), (s*0.26, s*0.04), (s*0.44, s*0.14)],
                 fill=(195, 110, 40))
    draw.polygon([(s*0.65, s*0.20), (s*0.72, s*0.04), (s*0.56, s*0.14)],
                 fill=(195, 110, 40))
    # Tail
    draw.arc([s*0.64, s*0.58, s*0.92, s*0.90], start=210, end=340,
             fill=(205, 118, 45), width=int(s*0.05))
    # Eyes
    draw.ellipse([s*0.39, s*0.26, s*0.47, s*0.34], fill=(155, 125, 28))
    draw.ellipse([s*0.54, s*0.26, s*0.62, s*0.34], fill=(155, 125, 28))
    # Paws
    draw.ellipse([s*0.30, s*0.80, s*0.46, s*0.90], fill=(235, 228, 215))
    draw.ellipse([s*0.56, s*0.80, s*0.72, s*0.90], fill=(235, 228, 215))

    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=85)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# robot_sketch.png
# Simulates: pencil-sketch line drawing of a robot/android on white.
# Tests luminance mode on a line-art image: dark lines on bright background.
# ---------------------------------------------------------------------------
def make_robot_sketch(size=240):
    import numpy as np
    rng = np.random.default_rng(99)
    noise = rng.integers(-4, 5, (size, size, 3), dtype=np.int16)
    arr = np.clip(np.full((size, size, 3), 254, dtype=np.int16) + noise, 240, 255)
    img = Image.fromarray(arr.astype(np.uint8), 'RGB')
    draw = ImageDraw.Draw(img)
    s = size
    ink = (22, 22, 22)
    lw = max(2, s // 70)

    # Head outline
    draw.ellipse([s*0.20, s*0.06, s*0.80, s*0.70], outline=ink, width=lw*2)
    # Robot skull plate
    draw.arc([s*0.22, s*0.08, s*0.78, s*0.42], start=200, end=340,
             fill=ink, width=lw)
    # Ear pieces
    draw.rectangle([int(s*0.76), int(s*0.32), int(s*0.86), int(s*0.52)],
                   outline=ink, width=lw)
    draw.rectangle([int(s*0.14), int(s*0.32), int(s*0.24), int(s*0.52)],
                   outline=ink, width=lw)
    # Eyes (outline + filled pupil)
    draw.ellipse([s*0.30, s*0.30, s*0.44, s*0.46], outline=ink, width=lw)
    draw.ellipse([s*0.56, s*0.30, s*0.70, s*0.46], outline=ink, width=lw)
    draw.ellipse([s*0.35, s*0.35, s*0.40, s*0.41], fill=ink)
    draw.ellipse([s*0.60, s*0.35, s*0.65, s*0.41], fill=ink)
    # Nose
    draw.line([(int(s*0.48), int(s*0.46)),
               (int(s*0.44), int(s*0.56)),
               (int(s*0.56), int(s*0.56))], fill=ink, width=lw)
    # Mouth
    draw.arc([s*0.35, s*0.52, s*0.65, s*0.66], start=10, end=170,
             fill=ink, width=lw)
    # Neck lines
    draw.line([(int(s*0.40), int(s*0.68)), (int(s*0.36), int(s*0.84))],
              fill=ink, width=lw)
    draw.line([(int(s*0.60), int(s*0.68)), (int(s*0.64), int(s*0.84))],
              fill=ink, width=lw)
    # Shoulders
    draw.arc([s*0.08, s*0.72, s*0.52, s*0.96], start=180, end=360,
             fill=ink, width=lw)
    draw.arc([s*0.48, s*0.72, s*0.92, s*0.96], start=180, end=360,
             fill=ink, width=lw)
    # Cross-hatch shading on sides of face
    for i in range(6):
        yo = s * (0.30 + i * 0.05)
        draw.line([(int(s*0.21), int(yo)), (int(s*0.34), int(yo + s*0.07))],
                  fill=(55, 55, 55), width=1)
        draw.line([(int(s*0.66), int(yo)), (int(s*0.79), int(yo + s*0.07))],
                  fill=(55, 55, 55), width=1)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Generating fixture images...')
    save(make_python_logo(200),        'python_logo_transparent.png')
    save(make_wikipedia_logo(200),     'wikipedia_logo_transparent.png')
    save(make_portrait(220),           'lena_grayscale.png')
    save(make_cat_photo(220),          'black_cat_photo.jpg',      fmt='JPEG', quality=85)
    save(make_dolphin_embroidery(240), 'dolphin_embroidery.jpg',   fmt='JPEG', quality=88)
    save(make_orange_cat_photo(240),   'orange_cat_photo.jpg',     fmt='JPEG', quality=85)
    save(make_robot_sketch(240),       'robot_sketch.png')
    print('Done.')
