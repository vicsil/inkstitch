# Authors: see git history
#
# Copyright (c) 2026 Authors
# Licensed under the GNU GPL version 3.0 or later.  See the file LICENSE for details.

"""Extension: Image to Satin Outline.

Traces the silhouette of a selected raster image and converts the resulting
outline path into an Ink/Stitch satin column element.

Works with:
  * Transparent PNGs (alpha-based outline detection)
  * Photos / JPEGs / opaque PNGs (luminance + Otsu-based detection)
"""

import base64
import os
from io import BytesIO
from itertools import chain
from urllib.parse import unquote

import inkex
from inkex import Boolean, errormsg
from inkex import Image as InkexImage
from PIL import Image
from shapely import geometry as shgeo

from ..elements.utils.stroke_to_satin import convert_path_to_satin, set_first_node
from ..i18n import _
from ..svg import PIXELS_PER_MM, get_correction_transform
from ..utils.smoothing import smooth_path
from shapely.geometry import JOIN_STYLE
from .base import InkstitchExtension
from .utils.image_outline import get_outline_coords


class ImageToSatinOutline(InkstitchExtension):
    """Trace a raster image outline and convert it to a satin stitch column."""

    def __init__(self, *args, **kwargs):
        InkstitchExtension.__init__(self, *args, **kwargs)
        self.arg_parser.add_argument('--notebook')
        self.arg_parser.add_argument(
            '-m', '--detection-mode', type=str, default='auto',
            dest='detection_mode',
            help='Outline detection mode: auto / alpha / luminance'
        )
        self.arg_parser.add_argument(
            '-t', '--threshold-adjust', type=int, default=0,
            dest='threshold_adjust',
            help='Offset added to the auto threshold (-50 … +50)'
        )
        self.arg_parser.add_argument(
            '-w', '--satin-width', type=float, default=2.0,
            dest='satin_width',
            help='Satin column width in mm'
        )
        self.arg_parser.add_argument(
            '-s', '--smoothness', type=float, default=0.3,
            dest='smoothness',
            help='Path smoothness in mm'
        )
        self.arg_parser.add_argument(
            '-p', '--simplification', type=float, default=1.0,
            dest='simplification',
            help='Contour simplification tolerance in mm'
        )
        self.arg_parser.add_argument(
            '-k', '--keep-original', type=Boolean, default=True,
            dest='keep_original',
            help='Keep the original image element'
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def effect(self):
        if not self.svg.selection:
            errormsg(_("Please select one or more image elements."))
            return

        processed = False
        for node in self.svg.selection:
            if not isinstance(node, InkexImage):
                errormsg(
                    _("{element_id} is not an image element. "
                      "Please select a raster image (PNG, JPEG, …).").format(
                        element_id=node.label or node.get_id())
                )
                continue
            if self._process_image(node):
                processed = True

        if not processed:
            errormsg(_("Could not generate a satin outline for any of the selected elements."))

    # ------------------------------------------------------------------
    # Per-image processing
    # ------------------------------------------------------------------

    def _process_image(self, node):
        """Process one SVG image node.  Returns True on success."""
        element_id = node.label or node.get_id()

        # Load PIL image
        pil_image = self._load_pil_image(node)
        if pil_image is None:
            errormsg(_("Could not load image data for {element_id}.").format(
                element_id=element_id))
            return False

        img_w, img_h = pil_image.size

        # Get SVG bounding box of the image element (in SVG user units / px)
        svg_x = float(node.get('x') or 0)
        svg_y = float(node.get('y') or 0)
        svg_w_str = node.get('width')
        svg_h_str = node.get('height')
        if not svg_w_str or not svg_h_str:
            errormsg(_("Image element {element_id} has no width/height attribute.").format(
                element_id=element_id))
            return False

        svg_w = self.svg.unittouu(svg_w_str)
        svg_h = self.svg.unittouu(svg_h_str)

        if svg_w == 0 or svg_h == 0:
            errormsg(_("Image element {element_id} has zero dimensions.").format(
                element_id=element_id))
            return False

        # Convert mm parameters to pixels
        satin_width_px = self.options.satin_width * PIXELS_PER_MM
        smoothness_px = self.options.smoothness * PIXELS_PER_MM
        simplification_px = self.options.simplification * PIXELS_PER_MM

        # Map simplification from SVG-px to image-px
        scale_x = img_w / svg_w
        simplification_img_px = simplification_px * scale_x

        # Get outline in image-pixel coordinates
        pixel_coords = get_outline_coords(
            pil_image,
            mode=self.options.detection_mode,
            threshold_adjust=self.options.threshold_adjust,
            simplification=simplification_img_px,
        )

        if not pixel_coords or len(pixel_coords) < 3:
            errormsg(_("Could not detect an outline for {element_id}. "
                       "Try adjusting the threshold or detection mode.").format(
                element_id=element_id))
            return False

        # Map from image pixels → SVG user units
        svg_coords = [
            (svg_x + px * (svg_w / img_w),
             svg_y + py * (svg_h / img_h))
            for px, py in pixel_coords
        ]

        # Smooth the contour
        smoothed = smooth_path(svg_coords, smoothness=smoothness_px)
        smoothed_tuples = [(p.x, p.y) for p in smoothed]

        if len(smoothed_tuples) < 3:
            errormsg(_("Smoothed outline for {element_id} is too short.").format(
                element_id=element_id))
            return False

        # Ensure the path is closed
        if smoothed_tuples[0] != smoothed_tuples[-1]:
            smoothed_tuples.append(smoothed_tuples[0])

        # Find a clean starting node for satin conversion (avoids rail self-intersection)
        paths_for_satin = [list(smoothed_tuples)]
        set_first_node(paths_for_satin, satin_width_px)
        final_path = paths_for_satin[0]

        # Convert to satin column
        style_args = {'join_style': JOIN_STYLE.round}
        satin_result = convert_path_to_satin(final_path, satin_width_px, style_args)

        if satin_result is None:
            errormsg(_("Could not convert outline to satin for {element_id}.").format(
                element_id=element_id))
            return False

        rails, rungs = satin_result

        # Filter rungs to only those that intersect both rails cleanly
        rungs = self._filter_rungs(rails, rungs)

        # Build the SVG path element
        path_el = self._build_satin_element(rails, rungs, node)

        # Insert into document (as sibling of the image, before it in z-order)
        parent = node.getparent()
        parent.insert(parent.index(node), path_el)

        if not self.options.keep_original:
            node.delete()

        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_pil_image(self, node):
        """Load a PIL Image from an SVG <image> element (embedded or linked)."""
        xlink = node.get('xlink:href') or node.get('href')
        if not xlink:
            return None

        try:
            if xlink.startswith('file'):
                path = unquote(xlink[7:])
                if not os.path.isfile(path):
                    return None
                img = Image.open(path)
            elif xlink.startswith('data'):
                _, data = xlink.split(',', 1)
                img = Image.open(BytesIO(base64.b64decode(data)))
            else:
                # Relative file path
                path = xlink
                if not os.path.isfile(path):
                    return None
                img = Image.open(path)

            # Convert to RGBA so downstream code always has 4 channels
            return img.convert('RGBA')
        except Exception:
            return None

    def _filter_rungs(self, rails, rungs):
        """Keep only rungs that intersect both rails at exactly two points."""
        rails_geom = shgeo.MultiLineString(rails)
        filtered = []
        for rung in shgeo.MultiLineString(rungs).geoms:
            intersection = rung.intersection(rails_geom)
            if (intersection.geom_type == 'MultiPoint'
                    and len(intersection.geoms) == 2):
                filtered.append(list(rung.coords))
        return filtered

    def _build_satin_element(self, rails, rungs, reference_node):
        """Create an inkex PathElement marked as a satin column."""
        d = ''
        for path in chain(rails, rungs):
            d += 'M'
            for x, y in path:
                d += '%s,%s ' % (x, y)
            d += ' '

        path_el = inkex.PathElement(attrib={'d': d})
        path_el.set('inkstitch:satin_column', True)
        path_el.set('id', self.uniqueId('path'))
        path_el.set('transform', get_correction_transform(reference_node))
        path_el.set('style', 'stroke:#000000;stroke-width:1px;fill:none')
        return path_el
