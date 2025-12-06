import cv2
import json
import argparse
import numpy as np
from math import ceil

###############################################################################
# Utility functions
###############################################################################

def detect_redactions(image_path):
    """Detects redaction bounding boxes and returns list of (x, y, w, h)."""
    img = cv2.imread(image_path)
    h_img, w_img = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Morphology tuned to your dataset
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Ignore tiny specks
        if w * h > 2000:
            boxes.append((x, y, w, h))

    return boxes, (w_img, h_img)


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


###############################################################################
# Main CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input PNG")
    parser.add_argument("--format", choices=["yolo", "coco"], default="yolo",
                        help="Annotation format to output")
    parser.add_argument("--out", default="annotations",
                        help="Output filename without extension")
    args = parser.parse_args()

    boxes, img_size = detect_redactions(args.image)

    typical_height, unit_count = estimate_redaction_units(boxes)

    print(f"Detected {len(boxes)} redaction regions.")
    print(f"Estimated typical line height: {typical_height:.2f}px")
    print(f"Estimated line-equivalent redaction count: {unit_count}")

    if args.format == "yolo":
        out_path = args.out + ".txt"
        save_yolo(boxes, img_size, out_path)
        print(f"YOLO annotations saved to {out_path}")

    else:
        out_path = args.out + ".json"
        save_coco(boxes, img_size, out_path, args.image)
        print(f"COCO annotations saved to {out_path}")


if __name__ == "__main__":
    main()
