import cv2
import numpy as np
import sys
import os
import math

def detect_redactions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Threshold (invert: black=255, white=0) ---
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # --- Morphological operations ---
    # Close small gaps within bars
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Open to separate bars that touch
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    # --- Find contours ---
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    redaction_boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # ignore small noise
        if area < 3000:
            continue

        # wide horizontal rectangles
        aspect = w / float(h)
        if aspect < 3:
            continue

        # dark uniform patch?
        patch = gray[y:y+h, x:x+w]
        if np.mean(patch) > 80:  # should be dark
            continue

        redaction_boxes.append((x, y, w, h))

    # Extract heights
    heights = [h for (_, _, _, h) in redaction_boxes]

    # Estimate typical redaction height (single line)
    # Use only small heights to avoid thick blocks
    small_heights = [h for h in heights if h < 50]

    if len(small_heights) > 0:
        typical_height = int(np.median(small_heights))
    else:
        typical_height = int(np.median(heights))  # fallback

    # print(f"Estimated typical redaction line height: {typical_height}")

    # Compute adjusted line count
    adjusted_count = 0
    details = []

    for (x, y, w, h) in redaction_boxes:
        lines = math.ceil(h / typical_height)
        adjusted_count += lines
        details.append((h, lines))

    # print("\nRedaction block analysis:")
    # for h, lines in details:
    #     print(f"  height={h} → {lines} line(s)")

    # print(f"\nAdjusted estimated number of redacted lines: {adjusted_count}")

    return img, redaction_boxes


def main():
    if len(sys.argv) != 2:
        print("Usage: python detect_redactions_v2.py <image.png>")
        sys.exit(1)

    image_path = sys.argv[1]

    img, redactions = detect_redactions(image_path)
    count = len(redactions)

    print(f"Detected redactions: {count}")

    # Draw boxes
    preview = img.copy()
    for (x, y, w, h) in redactions:
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save
    base, _ = os.path.splitext(image_path)
    out_path = base + "_redactions_preview.png"
    cv2.imwrite(out_path, preview)

    print(f"Saved preview to: {out_path}")


if __name__ == "__main__":
    main()
