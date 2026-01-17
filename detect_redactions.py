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

    metadata = {
        "estimated_dpi": round(dpi, 1),
        "skew_angle": round(skew_angle, 2),
        "threshold": int(threshold_val),
        "kernel_close_w": kernel_close_w,
        "min_area": min_area,
        "aggressive": aggressive
    }

    return boxes, img_size, gray, img, metadata


def filter_false_positives(boxes, gray_img, img_size, dark_threshold=50, min_dark_ratio=0.05,
                           min_aspect_ratio=2.0):
    """Filter out obvious false positives before confidence scoring.

    Removes:
    - Detections touching image edges (scanner borders)
    - Detections with aspect ratio < min_aspect_ratio (too square/vertical)
    - Detections with insufficient dark pixels (less than min_dark_ratio of pixels below dark_threshold)

    Args:
        boxes: List of (x, y, w, h) tuples
        gray_img: Grayscale image
        img_size: (width, height) of image
        dark_threshold: Pixel value below which is considered "dark"
        min_dark_ratio: Minimum fraction of dark pixels required
        min_aspect_ratio: Minimum width/height ratio (0.0 disables check for aggressive mode)
    """
    w_img, h_img = img_size
    filtered = []

    for box in boxes:
        x, y, w, h = box

        # Check if touching edge (scanner border artifact)
        edge_dist = min(x, y, w_img - (x + w), h_img - (y + h))
        if edge_dist <= 0:
            continue

        # Check aspect ratio (redactions are typically horizontal bars)
        # In aggressive mode (min_aspect_ratio=0), skip this check
        if min_aspect_ratio > 0:
            aspect_ratio = w / h
            if aspect_ratio < min_aspect_ratio:
                continue

        # Check if region contains sufficient dark pixels
        # This is better than mean pixel value because redactions mixed with
        # text will have high mean but still contain dark redaction pixels
        region = gray_img[y:y+h, x:x+w]
        dark_pixels = np.sum(region < dark_threshold)
        total_pixels = region.size
        dark_ratio = dark_pixels / total_pixels
        if dark_ratio < min_dark_ratio:
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


def save_json(detections, img_size, image_path, preview_path, coverage, metadata, out_path):
    """Saves JSON output with detections, confidence scores, coverage, and processing metadata."""
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
            # In aggressive mode: disable aspect ratio filter, lower dark ratio threshold
            min_aspect = 0.0 if aggressive else 2.0
            min_dark = 0.02 if aggressive else 0.05
            boxes = filter_false_positives(boxes, gray, img_size,
                                           min_aspect_ratio=min_aspect,
                                           min_dark_ratio=min_dark)
            if len(boxes) == 0:
                return {
                    "image": str(image_path),
                    "status": "all_filtered",
                    "detections": 0,
                    "filtered": raw_count,
                    "coverage_percent": 0.0,
                    "error": None
                }

        # Build detections with confidence
        detections = []
        for i, box in enumerate(boxes):
            confidence, factors = compute_confidence(box, gray, img_size, aggressive=aggressive)
            detections.append({
                "id": i + 1,
                "bbox": list(box),
                "confidence": round(confidence, 3),
                "confidence_factors": factors,
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

        # Calculate stats
        typical_height, unit_count = estimate_redaction_units(boxes)
        coverage = calculate_coverage(boxes, img_size)

        return {
            "image": str(image_path),
            "status": "success",
            "detections": len(detections),
            "filtered": raw_count - len(boxes) if not args.no_filter else 0,
            "coverage_percent": coverage["coverage_fraction"] * 100,
            "coverage": coverage,
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

    # Save detailed summary JSON
    summary_json = output_path / "batch_summary.json"
    summary_data = {
        "input_directory": str(input_path),
        "output_directory": str(output_path),
        "total_images": len(image_files),
        "successful": success_count,
        "total_detections": total_detections,
        "results": [{
            "relative_path": r.get("relative_path", r["image"]),
            "status": r["status"],
            "detections": r["detections"],
            "coverage_percent": round(r["coverage_percent"], 2),
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
        # In aggressive mode: disable aspect ratio filter, lower dark ratio threshold
        min_aspect = 0.0 if args.aggressive else 2.0
        min_dark = 0.02 if args.aggressive else 0.05
        boxes = filter_false_positives(boxes, gray, img_size,
                                       min_aspect_ratio=min_aspect,
                                       min_dark_ratio=min_dark)
        if len(boxes) == 0:
            print(f"No redaction regions detected ({raw_count} candidates filtered out).")
            return
        if len(boxes) < raw_count:
            print(f"Filtered {raw_count - len(boxes)} false positives.")

    typical_height, unit_count = estimate_redaction_units(boxes)

    # Build detections with confidence scores
    detections = []
    for i, box in enumerate(boxes):
        confidence, factors = compute_confidence(box, gray, img_size, aggressive=args.aggressive)
        detections.append({
            "id": i + 1,
            "bbox": list(box),
            "confidence": round(confidence, 3),
            "confidence_factors": factors,
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

    print(f"Detected {len(boxes)} redaction regions.")
    print(f"Estimated typical line height: {typical_height:.2f}px")
    print(f"Estimated line-equivalent redaction count: {unit_count}")
    print(f"Redaction coverage: {coverage['coverage_percent']} of content area")

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
        save_json(detections, img_size, args.input, preview_path, coverage, metadata, out_path)
        print(f"JSON annotations saved to {out_path}")


if __name__ == "__main__":
    main()
