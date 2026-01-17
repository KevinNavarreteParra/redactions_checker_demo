import cv2
import json
import argparse
import numpy as np
import os
import csv
from pathlib import Path
from math import ceil

###############################################################################
# Utility functions
###############################################################################

# Reference parameters calibrated for 300 DPI
BASE_DPI = 300
BASE_KERNEL_CLOSE_W = 50  # Horizontal closing kernel width
BASE_KERNEL_OPEN_W = 3    # Opening kernel width
BASE_KERNEL_OPEN_H = 9    # Opening kernel height
BASE_MIN_AREA = 2000      # Minimum contour area (normal mode)
BASE_MIN_AREA_AGGRESSIVE = 500  # Minimum contour area (aggressive mode)
BASE_KERNEL_SQUARE = 15   # Small square kernel for isolated words (aggressive mode)
BASE_MIN_HEIGHT = 10      # Minimum redaction height in pixels (at 300 DPI)


def estimate_dpi(img_size, page_width_inches=8.5, page_height_inches=11):
    """Estimate DPI assuming US Letter page size."""
    w_img, h_img = img_size
    dpi_from_width = w_img / page_width_inches
    dpi_from_height = h_img / page_height_inches
    # Use average, biased toward the larger dimension
    return (dpi_from_width + dpi_from_height) / 2


def deskew_image(img, gray):
    """Detect and correct document skew using Hough line detection."""
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return img, gray, 0.0

    # Find angles of horizontal-ish lines (within 45 degrees of horizontal)
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90  # Convert to deviation from horizontal
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return img, gray, 0.0

    # Use median angle to avoid outliers
    skew_angle = np.median(angles)

    # Only correct if skew is significant (> 0.5 degrees)
    if abs(skew_angle) < 0.5:
        return img, gray, 0.0

    # Rotate image to correct skew
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
    rotated_gray = cv2.warpAffine(gray, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

    return rotated_img, rotated_gray, skew_angle


def detect_solid_black_regions(gray, min_area=500):
    """Detect solid black regions directly without morphological merging.

    This catches small inline redactions that would otherwise be merged with
    surrounding text by the morphological operations.

    Args:
        gray: Grayscale image
        min_area: Minimum contour area

    Returns:
        List of (x, y, w, h) tuples for solid black regions
    """
    # Use strict threshold to find only very dark pixels
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        # Check if this is actually a solid black region
        region = gray[y:y+h, x:x+w]
        mean_val = np.mean(region)
        dark_ratio = np.sum(region < 50) / region.size

        # Must be very dark (mean < 30) and mostly solid (dark_ratio > 0.7)
        if mean_val < 30 and dark_ratio > 0.7:
            boxes.append((x, y, w, h))

    return boxes


def detect_redactions(image_path, adaptive=False, deskew=False, min_area_override=None, aggressive=False):
    """Detects redaction bounding boxes.

    Args:
        image_path: Path to input image
        adaptive: Use Otsu's method for automatic thresholding
        deskew: Correct document skew before processing
        min_area_override: Override the minimum contour area (default: auto-scaled)
        aggressive: Enable maximum recall mode for batch pre-processing

    Returns:
        boxes: List of (x, y, w, h) tuples
        img_size: (width, height) of image
        gray: Grayscale image
        img: Original (or deskewed) color image
        metadata: Dict with processing info (dpi, skew_angle, threshold)
    """
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]
    img_size = (w_img, h_img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional deskew
    skew_angle = 0.0
    if deskew:
        img, gray, skew_angle = deskew_image(img, gray)

    # Estimate DPI and scale parameters
    dpi = estimate_dpi(img_size)
    scale = dpi / BASE_DPI

    kernel_close_w = max(10, int(BASE_KERNEL_CLOSE_W * scale))
    kernel_open_w = max(2, int(BASE_KERNEL_OPEN_W * scale))
    kernel_open_h = max(3, int(BASE_KERNEL_OPEN_H * scale))

    # Scale min_area based on mode
    if min_area_override is not None:
        min_area = min_area_override
    elif aggressive:
        min_area = max(200, int(BASE_MIN_AREA_AGGRESSIVE * scale))
    else:
        min_area = max(300, int(BASE_MIN_AREA * scale))

    # Thresholding
    if adaptive:
        # Otsu's method finds optimal threshold automatically
        threshold_val, binary = cv2.threshold(gray, 0, 255,
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        threshold_val = 50
        _, binary = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Morphology with scaled kernels
    # Horizontal closing kernel (joins horizontally adjacent redactions)
    kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_close_w, 1))
    closed_h = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_h)

    if aggressive:
        # Add small square kernel to catch isolated word-level redactions
        kernel_sq_size = max(5, int(BASE_KERNEL_SQUARE * scale))
        kernel_close_sq = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_sq_size, kernel_sq_size))
        closed_sq = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close_sq)
        # Union both closed images
        closed = cv2.bitwise_or(closed_h, closed_sq)
    else:
        closed = closed_h

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_open_w, kernel_open_h))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > min_area:
            boxes.append((x, y, w, h))

    # Also detect solid black regions directly (catches small inline redactions)
    # These are added directly - the strict filter will later remove text-line
    # morphology boxes while keeping solid black boxes
    solid_black_boxes = detect_solid_black_regions(gray, min_area=min_area // 2)
    boxes.extend(solid_black_boxes)

    metadata = {
        "estimated_dpi": round(dpi, 1),
        "skew_angle": round(skew_angle, 2),
        "threshold": int(threshold_val),
        "kernel_close_w": kernel_close_w,
        "min_area": min_area,
        "aggressive": aggressive
    }

    return boxes, img_size, gray, img, metadata


def deduplicate_boxes(boxes, iou_threshold=0.5):
    """Remove duplicate/overlapping boxes, keeping the larger one.

    Args:
        boxes: List of (x, y, w, h) tuples
        iou_threshold: Boxes with IoU above this are considered duplicates

    Returns:
        Deduplicated list of boxes
    """
    if len(boxes) <= 1:
        return boxes

    # Sort by area (largest first) so we keep larger boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []

    for box in boxes:
        x1, y1, w1, h1 = box
        is_duplicate = False

        for kept in keep:
            x2, y2, w2, h2 = kept

            # Calculate IoU
            inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            inter_area = inter_x * inter_y

            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - inter_area

            iou = inter_area / union_area if union_area > 0 else 0

            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            keep.append(box)

    return keep


def filter_false_positives(boxes, gray_img, img_size, color_img=None, scale=1.0,
                           dark_threshold=50, min_dark_ratio=0.05,
                           min_aspect_ratio=2.0, strict=True):
    """Filter out obvious false positives before confidence scoring.

    Removes:
    - Detections touching image edges (scanner borders)
    - Detections with aspect ratio < min_aspect_ratio (too square/vertical)
    - Detections with insufficient dark pixels (less than min_dark_ratio of pixels below dark_threshold)
    - In strict mode: detections that don't look like solid black fills (text lines)
    - Colored (non-black) regions based on saturation (if color_img provided)
    - Thin lines below minimum height threshold (underlines, separators)

    Args:
        boxes: List of (x, y, w, h) tuples
        gray_img: Grayscale image
        img_size: (width, height) of image
        color_img: Original color image (BGR) for saturation check
        scale: DPI scale factor for height threshold calculation
        dark_threshold: Pixel value below which is considered "dark"
        min_dark_ratio: Minimum fraction of dark pixels required
        min_aspect_ratio: Minimum width/height ratio (0.0 disables check for aggressive mode)
        strict: If True, require solid black fill characteristics (filters out text lines)
    """
    w_img, h_img = img_size
    filtered = []

    # Calculate minimum height threshold (scales with DPI)
    # Real redactions cover at least part of a text line (~15-30px at 200-300 DPI)
    # Underlines are typically 1-3px
    min_height = max(5, int(BASE_MIN_HEIGHT * scale))

    for box in boxes:
        x, y, w, h = box

        # Check if touching edge (scanner border artifact)
        edge_dist = min(x, y, w_img - (x + w), h_img - (y + h))
        if edge_dist <= 0:
            continue

        # Check minimum height (filter out thin underlines and separators)
        if h < min_height:
            continue

        # Check color saturation to filter out colored (non-black) lines
        # Black/gray has low saturation in HSV; red/blue/green have high saturation
        if color_img is not None:
            region_color = color_img[y:y+h, x:x+w]
            hsv = cv2.cvtColor(region_color, cv2.COLOR_BGR2HSV)
            mean_saturation = np.mean(hsv[:, :, 1])
            if mean_saturation > 40:  # Threshold: black/gray < 30, colors > 40
                continue

        # Extract region for analysis
        region = gray_img[y:y+h, x:x+w]
        mean_val = np.mean(region)
        dark_pixels = np.sum(region < dark_threshold)
        total_pixels = region.size
        dark_ratio = dark_pixels / total_pixels

        # Skip aspect ratio check for confirmed solid black boxes
        # (they're definitely redactions regardless of shape)
        is_confirmed_solid = mean_val < 30 and dark_ratio > 0.8

        # Check aspect ratio (redactions are typically horizontal bars)
        # In aggressive mode (min_aspect_ratio=0), skip this check
        if min_aspect_ratio > 0 and not is_confirmed_solid:
            aspect_ratio = w / h
            if aspect_ratio < min_aspect_ratio:
                continue

        # Check if region contains sufficient dark pixels
        if dark_ratio < min_dark_ratio:
            continue

        # Strict mode: require solid black fill characteristics
        # This filters out text lines which have high mean, high std, low dark_ratio
        if strict:
            std_val = np.std(region)

            # Real redactions (solid black boxes) have:
            # - Low mean pixel value - mostly black
            # - Low standard deviation - uniform fill, not text
            # - High dark ratio - majority of pixels are dark
            # Text lines typically have mean ~180-220, std ~80-100, dark_ratio ~0.1-0.2

            # Use tiered approach: very confident if all criteria met,
            # but also accept if strongly meets some criteria
            is_very_dark = mean_val < 50  # Nearly solid black
            is_high_dark_ratio = dark_ratio > 0.6  # Mostly dark pixels
            is_uniform = std_val < 50  # Low variation (solid fill)

            # Accept if: very dark, OR high dark ratio with reasonable mean,
            # OR meets relaxed combined thresholds
            is_solid_black = (
                is_very_dark or
                (is_high_dark_ratio and mean_val < 120) or
                (mean_val < 100 and std_val < 70 and dark_ratio > 0.4)
            )

            if not is_solid_black:
                continue

        filtered.append(box)

    return filtered


def compute_confidence(box, gray_img, img_size, dark_threshold=50, aggressive=False):
    """Compute 0-1 confidence score for a detection.

    Args:
        box: (x, y, w, h) tuple
        gray_img: Grayscale image
        img_size: (width, height) of image
        dark_threshold: Pixel value below which is considered "dark"
        aggressive: If True, use relaxed thresholds for inline/word-level redactions
    """
    x, y, w, h = box
    w_img, h_img = img_size

    # Extract region from grayscale image
    region = gray_img[y:y+h, x:x+w]
    mean_val = np.mean(region)
    std_val = np.std(region)
    aspect_ratio = w / h
    edge_dist = min(x, y, w_img - (x + w), h_img - (y + h))

    # Calculate dark pixel ratio (percentage of pixels below threshold)
    dark_pixels = np.sum(region < dark_threshold)
    total_pixels = region.size
    dark_ratio = dark_pixels / total_pixels

    # Start with 1.0 and penalize suspicious characteristics
    confidence = 1.0

    # Dark ratio is the most important factor - higher is better
    if dark_ratio < 0.1:
        confidence -= 0.3
    elif dark_ratio < 0.2:
        confidence -= 0.15
    elif dark_ratio > 0.5:
        confidence += 0.1  # Bonus for very dark regions

    # Aspect ratio checks - relaxed in aggressive mode
    if aggressive:
        # Only penalize extreme aspect ratios (very tall/narrow or very wide)
        if aspect_ratio < 0.5 or aspect_ratio > 100:
            confidence -= 0.15
    else:
        if aspect_ratio < 3 or aspect_ratio > 50:
            confidence -= 0.25

    # Edge distance check
    if edge_dist < 20:
        confidence -= 0.3

    return max(0.0, min(1.0, confidence)), {
        "mean_pixel": float(mean_val),
        "std_dev": float(std_val),
        "aspect_ratio": float(aspect_ratio),
        "edge_distance": int(edge_dist),
        "dark_ratio": float(dark_ratio)
    }


def estimate_redaction_units(boxes):
    """Estimates typical single-line height and computes line-equivalent counts."""
    heights = np.array([h for (_, _, _, h) in boxes])

    # Remove extreme outliers (like full-page black)
    filtered = heights[(heights > 10) & (heights < np.percentile(heights, 95))]

    if len(filtered) == 0:
        return heights.mean(), len(heights)

    typical_height = np.median(filtered)

    # Each box contributes ceil(h / typical)
    total_units = sum(ceil(h / typical_height) for h in heights)

    return typical_height, total_units


def calculate_coverage(boxes, img_size, margin_fraction=0.1):
    """Calculate area-based redaction coverage.

    Returns coverage as fraction of estimated content area (excluding margins).
    """
    w_img, h_img = img_size

    # Estimate content area (excluding margins on all sides)
    content_width = w_img * (1 - 2 * margin_fraction)
    content_height = h_img * (1 - 2 * margin_fraction)
    content_area = content_width * content_height

    # Sum redaction areas
    total_redaction_area = sum(w * h for (x, y, w, h) in boxes)

    # Calculate coverage (cap at 1.0 since boxes may overlap)
    coverage = total_redaction_area / max(content_area, 1)

    return {
        "redaction_area_px": total_redaction_area,
        "content_area_px": int(content_area),
        "coverage_fraction": min(coverage, 1.0),
        "coverage_percent": f"{min(coverage * 100, 100):.1f}%"
    }


def classify_redaction(box, content_width, typical_height):
    """Classify a single redaction by type based on size.

    Args:
        box: (x, y, w, h) tuple
        content_width: Estimated content width (excluding margins)
        typical_height: Typical single-line height

    Returns:
        One of: "inline", "partial_line", "full_line", "block"
    """
    x, y, w, h = box
    width_ratio = w / content_width if content_width > 0 else 0
    height_ratio = h / typical_height if typical_height > 0 else 1

    if height_ratio > 1.5:
        return "block"
    elif width_ratio > 0.70:
        return "full_line"
    elif width_ratio > 0.15:
        return "partial_line"
    else:
        return "inline"


def get_vertical_zone(y_center, img_height):
    """Determine vertical zone for a detection.

    Args:
        y_center: Y-coordinate of detection center
        img_height: Total image height

    Returns:
        One of: "header", "body", "footer"
    """
    relative_y = y_center / img_height if img_height > 0 else 0.5
    if relative_y < 0.15:
        return "header"
    elif relative_y > 0.85:
        return "footer"
    else:
        return "body"


def compute_derived_metrics(boxes, img_size, typical_height, coverage_fraction):
    """Compute all derived metrics for a document.

    Args:
        boxes: List of (x, y, w, h) tuples
        img_size: (width, height) of image
        typical_height: Typical single-line height from estimate_redaction_units
        coverage_fraction: Coverage as fraction (0-1) from calculate_coverage

    Returns:
        Dict with classification, spatial, and intensity sections
    """
    w_img, h_img = img_size
    margin_fraction = 0.1
    content_width = w_img * (1 - 2 * margin_fraction)

    # Classification counts
    type_counts = {"inline": 0, "partial_line": 0, "full_line": 0, "block": 0}

    # Spatial counts
    zone_counts = {"header": 0, "body": 0, "footer": 0}
    y_centers = []

    for box in boxes:
        x, y, w, h = box
        y_center = y + h / 2

        # Classify redaction type
        redaction_type = classify_redaction(box, content_width, typical_height)
        type_counts[redaction_type] += 1

        # Classify vertical zone
        zone = get_vertical_zone(y_center, h_img)
        zone_counts[zone] += 1

        y_centers.append(y_center)

    # Calculate vertical spread (normalized std dev of y-positions)
    if len(y_centers) > 1:
        y_std = np.std(y_centers)
        vertical_spread = min(y_std / h_img, 1.0) if h_img > 0 else 0.0
    else:
        vertical_spread = 0.0

    # Document-level intensity classification
    detection_count = len(boxes)
    coverage_percent = coverage_fraction * 100
    block_count = type_counts["block"]

    if detection_count == 0:
        redaction_level = "none"
    elif detection_count <= 2 and coverage_percent < 1.0:
        redaction_level = "light"
    elif detection_count > 10 or coverage_percent > 5.0:
        redaction_level = "heavy"
    else:
        redaction_level = "moderate"

    # Heaviness score (0-100)
    # 40% from detection count (normalized to 0-20 range)
    count_score = min(detection_count / 20, 1.0) * 40
    # 40% from coverage percentage (normalized to 0-10% range)
    coverage_score = min(coverage_percent / 10, 1.0) * 40
    # 20% from block count (multi-line redactions weighted higher)
    block_score = min(block_count / 5, 1.0) * 20

    heaviness_score = round(count_score + coverage_score + block_score)

    return {
        "classification": {
            "inline_count": type_counts["inline"],
            "partial_line_count": type_counts["partial_line"],
            "full_line_count": type_counts["full_line"],
            "block_count": type_counts["block"]
        },
        "spatial": {
            "header_count": zone_counts["header"],
            "body_count": zone_counts["body"],
            "footer_count": zone_counts["footer"],
            "vertical_spread": round(vertical_spread, 3)
        },
        "intensity": {
            "redaction_level": redaction_level,
            "heaviness_score": heaviness_score
        }
    }


###############################################################################
# Output formats
###############################################################################

def save_yolo(boxes, img_size, out_path):
    """Saves YOLO annotation file with class=0 for 'redaction'."""
    w_img, h_img = img_size
    lines = []

    for (x, y, w, h) in boxes:
        xc = (x + w / 2) / w_img
        yc = (y + h / 2) / h_img
        wn = w / w_img
        hn = h / h_img

        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def save_coco(boxes, img_size, out_path, image_path):
    """Saves COCO annotation JSON (single image, single category)."""
    w_img, h_img = img_size

    coco = {
        "images": [
            {
                "id": 1,
                "file_name": image_path,
                "width": w_img,
                "height": h_img,
            }
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "redaction"}],
    }

    ann_id = 1
    for (x, y, w, h) in boxes:
        coco["annotations"].append({
            "id": ann_id,
            "image_id": 1,
            "category_id": 1,
            "bbox": [x, y, w, h],  # COCO = [x, y, width, height]
            "area": float(w * h),
            "iscrowd": 0
        })
        ann_id += 1

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)


def save_coco_batch(coco_data, out_path):
    """Saves consolidated COCO annotation JSON for all images in a batch.

    Args:
        coco_data: Dict with 'images' list and 'annotations' list collected during batch
        out_path: Output file path
    """
    coco = {
        "images": coco_data["images"],
        "annotations": coco_data["annotations"],
        "categories": [{"id": 1, "name": "redaction"}],
    }

    with open(out_path, "w") as f:
        json.dump(coco, f, indent=2)


def generate_preview(img, detections, out_path):
    """Generate preview image with color-coded bounding boxes."""
    preview = img.copy()
    for det in detections:
        conf = det["confidence"]
        if conf >= 0.8:
            color = (0, 255, 0)    # Green (BGR)
        elif conf >= 0.5:
            color = (0, 255, 255)  # Yellow (BGR)
        else:
            color = (0, 0, 255)    # Red (BGR)

        x, y, w, h = det["bbox"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite(out_path, preview)


def save_json(detections, img_size, image_path, preview_path, coverage, metadata, derived_metrics, out_path):
    """Saves JSON output with detections, confidence scores, coverage, derived metrics, and processing metadata."""
    # Count by confidence level
    high = sum(1 for d in detections if d["confidence"] >= 0.8)
    medium = sum(1 for d in detections if 0.5 <= d["confidence"] < 0.8)
    low = sum(1 for d in detections if d["confidence"] < 0.5)

    output = {
        "image_path": image_path,
        "image_size": list(img_size),
        "preview_path": preview_path,
        "processing": metadata,
        "detections": detections,
        "coverage": coverage,
        "classification": derived_metrics["classification"],
        "spatial": derived_metrics["spatial"],
        "intensity": derived_metrics["intensity"],
        "summary": {
            "total": len(detections),
            "high_confidence": high,
            "medium_confidence": medium,
            "low_confidence": low
        }
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)


###############################################################################
# Batch processing
###############################################################################

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}


def process_single_image(image_path, args):
    """Process a single image and return results dict (or None on failure)."""
    try:
        min_area = getattr(args, 'min_area', None)
        aggressive = getattr(args, 'aggressive', False)
        boxes, img_size, gray, img, metadata = detect_redactions(
            image_path, adaptive=args.adaptive, deskew=args.deskew,
            min_area_override=min_area, aggressive=aggressive
        )

        if len(boxes) == 0:
            return {
                "image": str(image_path),
                "status": "no_detections",
                "detections": 0,
                "coverage_percent": 0.0,
                "error": None
            }

        # Filter false positives
        raw_count = len(boxes)
        if not args.no_filter:
            # In aggressive mode: disable aspect ratio filter, lower dark ratio threshold, disable strict
            min_aspect = 0.0 if aggressive else 2.0
            min_dark = 0.02 if aggressive else 0.05
            strict = not aggressive  # Strict mode filters out text lines
            scale = metadata["estimated_dpi"] / BASE_DPI
            boxes = filter_false_positives(boxes, gray, img_size,
                                           color_img=img, scale=scale,
                                           min_aspect_ratio=min_aspect,
                                           min_dark_ratio=min_dark,
                                           strict=strict)
            # Deduplicate overlapping boxes
            boxes = deduplicate_boxes(boxes)
            if len(boxes) == 0:
                return {
                    "image": str(image_path),
                    "status": "all_filtered",
                    "detections": 0,
                    "filtered": raw_count,
                    "coverage_percent": 0.0,
                    "error": None
                }

        # Calculate stats needed for per-detection classification
        typical_height, unit_count = estimate_redaction_units(boxes)
        w_img, h_img = img_size
        margin_fraction = 0.1
        content_width = w_img * (1 - 2 * margin_fraction)

        # Build detections with confidence and per-detection derived fields
        detections = []
        for i, box in enumerate(boxes):
            x, y, w, h = box
            y_center = y + h / 2
            confidence, factors = compute_confidence(box, gray, img_size, aggressive=aggressive)
            detections.append({
                "id": i + 1,
                "bbox": list(box),
                "confidence": round(confidence, 3),
                "confidence_factors": factors,
                "redaction_type": classify_redaction(box, content_width, typical_height),
                "vertical_zone": get_vertical_zone(y_center, h_img),
                "status": "pending"
            })

        # Apply min-confidence filter
        if args.min_confidence > 0:
            filtered_pairs = [(d, b) for d, b in zip(detections, boxes)
                              if d["confidence"] >= args.min_confidence]
            if filtered_pairs:
                detections, boxes = zip(*filtered_pairs)
                detections, boxes = list(detections), list(boxes)
            else:
                detections, boxes = [], []

        if len(detections) == 0:
            return {
                "image": str(image_path),
                "status": "confidence_filtered",
                "detections": 0,
                "coverage_percent": 0.0,
                "error": None
            }

        # Calculate coverage and derived metrics
        coverage = calculate_coverage(boxes, img_size)
        derived_metrics = compute_derived_metrics(boxes, img_size, typical_height, coverage["coverage_fraction"])

        return {
            "image": str(image_path),
            "status": "success",
            "detections": len(detections),
            "filtered": raw_count - len(boxes) if not args.no_filter else 0,
            "coverage_percent": coverage["coverage_fraction"] * 100,
            "coverage": coverage,
            "derived_metrics": derived_metrics,
            "typical_height": typical_height,
            "unit_count": unit_count,
            "metadata": metadata,
            "detection_list": detections,
            "boxes": boxes,
            "img_size": img_size,
            "img": img,
            "error": None
        }

    except Exception as e:
        return {
            "image": str(image_path),
            "status": "error",
            "detections": 0,
            "coverage_percent": 0.0,
            "error": str(e)
        }


def run_batch(input_dir, output_dir, args):
    """Process all images in a directory, preserving subdirectory structure."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files recursively (including subdirectories)
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_path.glob(f"**/*{ext}"))
        image_files.extend(input_path.glob(f"**/*{ext.upper()}"))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")
    print("-" * 60)

    results = []
    success_count = 0
    total_detections = 0

    # For COCO format: collect all annotations into single file
    coco_data = {"images": [], "annotations": []}
    coco_image_id = 0
    coco_ann_id = 0

    for i, image_file in enumerate(image_files, 1):
        # Calculate relative path from input directory to preserve structure
        relative_path = image_file.relative_to(input_path)
        relative_dir = relative_path.parent
        stem = image_file.stem

        # Create corresponding subdirectory in output
        out_subdir = output_path / relative_dir
        out_subdir.mkdir(parents=True, exist_ok=True)

        # Progress - show relative path for clarity
        print(f"[{i}/{len(image_files)}] Processing {relative_path}...", end=" ", flush=True)

        result = process_single_image(str(image_file), args)
        result["relative_path"] = str(relative_path)  # Store for summary
        results.append(result)

        if result["status"] == "success":
            success_count += 1
            total_detections += result["detections"]

            # Save individual outputs in corresponding subdirectory
            if args.format == "json":
                out_file = out_subdir / f"{stem}.json"
                preview_file = out_subdir / f"{stem}_preview.png"
                generate_preview(result["img"], result["detection_list"], str(preview_file))
                save_json(
                    result["detection_list"],
                    result["img_size"],
                    str(image_file),
                    str(preview_file),
                    result["coverage"],
                    result["metadata"],
                    result["derived_metrics"],
                    str(out_file)
                )
            elif args.format == "yolo":
                out_file = out_subdir / f"{stem}.txt"
                save_yolo(result["boxes"], result["img_size"], str(out_file))
            elif args.format == "coco":
                # Collect COCO data for consolidated output (saved after loop)
                coco_image_id += 1
                w_img, h_img = result["img_size"]
                coco_data["images"].append({
                    "id": coco_image_id,
                    "file_name": str(relative_path),
                    "width": w_img,
                    "height": h_img,
                })
                for (x, y, w, h) in result["boxes"]:
                    coco_ann_id += 1
                    coco_data["annotations"].append({
                        "id": coco_ann_id,
                        "image_id": coco_image_id,
                        "category_id": 1,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0
                    })

            # Optional preview for non-json formats
            if args.preview and args.format != "json":
                preview_file = out_subdir / f"{stem}_preview.png"
                generate_preview(result["img"], result["detection_list"], str(preview_file))

            print(f"{result['detections']} detections, {result['coverage_percent']:.1f}% coverage")
        elif result["status"] == "error":
            print(f"ERROR: {result['error']}")
        else:
            print(f"{result['status']}")

    # Save consolidated COCO file if using COCO format
    if args.format == "coco" and coco_data["images"]:
        coco_out_path = output_path / "annotations.json"
        save_coco_batch(coco_data, str(coco_out_path))
        print(f"\nCOCO annotations saved to {coco_out_path}")

    # Print summary
    print("-" * 60)
    print(f"Processed {len(image_files)} images:")
    print(f"  Success: {success_count}")
    print(f"  No detections: {sum(1 for r in results if r['status'] == 'no_detections')}")
    print(f"  All filtered: {sum(1 for r in results if r['status'] == 'all_filtered')}")
    print(f"  Errors: {sum(1 for r in results if r['status'] == 'error')}")
    print(f"  Total detections: {total_detections}")

    # Save summary CSV
    summary_csv = output_path / "batch_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "relative_path", "status", "detections", "coverage_percent", "error"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "relative_path": r.get("relative_path", r["image"]),
                "status": r["status"],
                "detections": r["detections"],
                "coverage_percent": round(r["coverage_percent"], 2),
                "error": r["error"] or ""
            })
    print(f"\nSummary saved to {summary_csv}")

    # Save detailed summary JSON with aggregate derived metrics
    summary_json = output_path / "batch_summary.json"

    # Compute aggregate derived metrics from successful results
    agg_classification = {"inline_count": 0, "partial_line_count": 0, "full_line_count": 0, "block_count": 0}
    agg_spatial = {"header_count": 0, "body_count": 0, "footer_count": 0}
    intensity_counts = {"none": 0, "light": 0, "moderate": 0, "heavy": 0}
    heaviness_scores = []

    for r in results:
        if r["status"] == "success" and "derived_metrics" in r:
            dm = r["derived_metrics"]
            # Aggregate classification counts
            for key in agg_classification:
                agg_classification[key] += dm["classification"].get(key, 0)
            # Aggregate spatial counts
            for key in agg_spatial:
                agg_spatial[key] += dm["spatial"].get(key, 0)
            # Count intensity levels
            level = dm["intensity"].get("redaction_level", "none")
            intensity_counts[level] = intensity_counts.get(level, 0) + 1
            heaviness_scores.append(dm["intensity"].get("heaviness_score", 0))
        elif r["status"] in ("no_detections", "all_filtered", "confidence_filtered"):
            intensity_counts["none"] += 1

    # Calculate average heaviness score
    avg_heaviness = round(sum(heaviness_scores) / len(heaviness_scores), 1) if heaviness_scores else 0

    summary_data = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "total_images": len(image_files),
        "successful": success_count,
        "total_detections": total_detections,
        "aggregate_classification": agg_classification,
        "aggregate_spatial": agg_spatial,
        "intensity_distribution": intensity_counts,
        "average_heaviness_score": avg_heaviness,
        "results": [{
            "relative_path": r.get("relative_path", r["image"]),
            "status": r["status"],
            "detections": r["detections"],
            "coverage_percent": round(r["coverage_percent"], 2),
            "redaction_level": r.get("derived_metrics", {}).get("intensity", {}).get("redaction_level", "none") if r["status"] == "success" else "none",
            "heaviness_score": r.get("derived_metrics", {}).get("intensity", {}).get("heaviness_score", 0) if r["status"] == "success" else 0,
            "error": r["error"]
        } for r in results]
    }
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Detailed summary saved to {summary_json}")


###############################################################################
# Main CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Detect redacted regions in document images"
    )
    parser.add_argument("input", help="Path to input image or directory (for batch mode)")
    parser.add_argument("--format", choices=["yolo", "coco", "json"], default="yolo",
                        help="Annotation format to output")
    parser.add_argument("--out", default="annotations",
                        help="Output filename (single file) or directory (batch mode)")
    parser.add_argument("--preview", action="store_true",
                        help="Generate preview image with bounding boxes")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable false positive filtering")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="Minimum confidence threshold (0-1) for output")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use Otsu's method for automatic thresholding")
    parser.add_argument("--deskew", action="store_true",
                        help="Correct document skew before processing")
    parser.add_argument("--min-area", type=int, default=None,
                        help="Minimum contour area in pixels (default: auto-scaled by DPI)")
    parser.add_argument("--aggressive", action="store_true",
                        help="Enable maximum recall mode: disable aspect ratio filter, lower min_area, "
                             "add square kernel for isolated word detection")
    args = parser.parse_args()

    # Check if input is a directory (batch mode)
    input_path = Path(args.input)
    if input_path.is_dir():
        output_dir = args.out if args.out != "annotations" else "batch_output"
        run_batch(args.input, output_dir, args)
        return

    # Single file mode
    boxes, img_size, gray, img, metadata = detect_redactions(
        args.input, adaptive=args.adaptive, deskew=args.deskew,
        min_area_override=args.min_area, aggressive=args.aggressive
    )

    # Print processing info
    print(f"Estimated DPI: {metadata['estimated_dpi']}")
    if metadata['skew_angle'] != 0:
        print(f"Corrected skew: {metadata['skew_angle']}°")
    if args.adaptive:
        print(f"Auto threshold (Otsu): {metadata['threshold']}")
    if args.aggressive:
        print("Aggressive mode: maximum recall settings enabled")

    if len(boxes) == 0:
        print("No redaction regions detected.")
        return

    # Filter obvious false positives unless disabled
    raw_count = len(boxes)
    if not args.no_filter:
        # In aggressive mode: disable aspect ratio filter, lower dark ratio threshold, disable strict
        min_aspect = 0.0 if args.aggressive else 2.0
        min_dark = 0.02 if args.aggressive else 0.05
        strict = not args.aggressive  # Strict mode filters out text lines
        scale = metadata["estimated_dpi"] / BASE_DPI
        boxes = filter_false_positives(boxes, gray, img_size,
                                       color_img=img, scale=scale,
                                       min_aspect_ratio=min_aspect,
                                       min_dark_ratio=min_dark,
                                       strict=strict)
        # Deduplicate overlapping boxes
        boxes = deduplicate_boxes(boxes)
        if len(boxes) == 0:
            print(f"No redaction regions detected ({raw_count} candidates filtered out).")
            return
        if len(boxes) < raw_count:
            print(f"Filtered {raw_count - len(boxes)} false positives.")

    typical_height, unit_count = estimate_redaction_units(boxes)

    # Calculate content width for classification
    w_img, h_img = img_size
    margin_fraction = 0.1
    content_width = w_img * (1 - 2 * margin_fraction)

    # Build detections with confidence scores and per-detection derived fields
    detections = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        y_center = y + h / 2
        confidence, factors = compute_confidence(box, gray, img_size, aggressive=args.aggressive)
        detections.append({
            "id": i + 1,
            "bbox": list(box),
            "confidence": round(confidence, 3),
            "confidence_factors": factors,
            "redaction_type": classify_redaction(box, content_width, typical_height),
            "vertical_zone": get_vertical_zone(y_center, h_img),
            "status": "pending"
        })

    # Apply min-confidence filter if specified
    if args.min_confidence > 0:
        pre_filter_count = len(detections)
        filtered_pairs = [(d, b) for d, b in zip(detections, boxes)
                          if d["confidence"] >= args.min_confidence]
        if filtered_pairs:
            detections, boxes = zip(*filtered_pairs)
            detections, boxes = list(detections), list(boxes)
        else:
            detections, boxes = [], []

        if len(detections) < pre_filter_count:
            print(f"Filtered {pre_filter_count - len(detections)} low-confidence detections (threshold: {args.min_confidence}).")

        if len(detections) == 0:
            print("No detections remaining after confidence filtering.")
            return

    # Calculate coverage
    coverage = calculate_coverage(boxes, img_size)

    # Compute derived metrics
    derived_metrics = compute_derived_metrics(boxes, img_size, typical_height, coverage["coverage_fraction"])

    print(f"Detected {len(boxes)} redaction regions.")
    print(f"Estimated typical line height: {typical_height:.2f}px")
    print(f"Estimated line-equivalent redaction count: {unit_count}")
    print(f"Redaction coverage: {coverage['coverage_percent']} of content area")
    print(f"Redaction intensity: {derived_metrics['intensity']['redaction_level']} (score: {derived_metrics['intensity']['heaviness_score']})")

    # Generate preview if requested or if json format
    preview_path = None
    if args.preview or args.format == "json":
        preview_path = args.out + "_preview.png"
        generate_preview(img, detections, preview_path)
        print(f"Preview saved to {preview_path}")

    # Save output in requested format
    if args.format == "yolo":
        out_path = args.out + ".txt"
        save_yolo(boxes, img_size, out_path)
        print(f"YOLO annotations saved to {out_path}")

    elif args.format == "coco":
        out_path = args.out + ".json"
        save_coco(boxes, img_size, out_path, args.input)
        print(f"COCO annotations saved to {out_path}")

    else:  # json
        out_path = args.out + ".json"
        save_json(detections, img_size, args.input, preview_path, coverage, metadata, derived_metrics, out_path)
        print(f"JSON annotations saved to {out_path}")


if __name__ == "__main__":
    main()
