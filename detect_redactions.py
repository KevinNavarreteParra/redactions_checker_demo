import cv2
import numpy as np
import sys
import os

def detect_redactions(image_path):
    # --- Load image ---
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Preprocess ---
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray_blur)

    # --- Threshold the image ---
    _, thresh = cv2.threshold(
        gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Find contours ---
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    redaction_boxes = []

    # --- Filter for redaction-like shapes ---
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # Ignore tiny artifacts
        if area < 500:
            continue

        # Redactions often have wide rectangles (tweakable)
        aspect_ratio = w / float(h)
        if aspect_ratio < 2:
            continue

        # Uniformity check: redactions are nearly solid
        patch = gray[y:y+h, x:x+w]
        if np.std(patch) > 15:
            continue

        redaction_boxes.append((x, y, w, h))

    return img, redaction_boxes


def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_redactions.py <image.png>")
        sys.exit(1)

    image_path = sys.argv[1]

    img, redactions = detect_redactions(image_path)
    count = len(redactions)

    print(f"Detected redactions: {count}")

    # Draw boxes on a copy
    preview = img.copy()
    for (x, y, w, h) in redactions:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save preview
    base, _ = os.path.splitext(image_path)
    out_path = base + "_redactions_preview.png"
    cv2.imwrite(out_path, preview)

    print(f"Saved preview to: {out_path}")


if __name__ == "__main__":
    main()
