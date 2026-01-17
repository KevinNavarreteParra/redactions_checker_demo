# Redaction Detector

A Python CLI tool that detects redacted (blacked-out) regions in document images using OpenCV morphological operations. Exports annotations in YOLO, COCO, or JSON format.

## Installation

Requires Python 3.13.2 or higher. Install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

Or with pip:

```bash
pip install numpy opencv-python
```

## Usage

### Single Image

```bash
# Basic usage (outputs YOLO format)
python detect_redactions.py image.png

# Output in JSON format with preview image
python detect_redactions.py image.png --format json --out results

# Output in COCO format
python detect_redactions.py image.png --format coco --out annotations
```

### Batch Processing (Directory)

```bash
# Process all images in a directory
python detect_redactions.py input_folder/ --format json --out output_folder/

# Process with preview images for non-JSON formats
python detect_redactions.py input_folder/ --format yolo --out output_folder/ --preview
```

## Output Formats

| Format | Description | Output Files |
|--------|-------------|--------------|
| `yolo` | YOLO annotation format (normalized coordinates) | `*.txt` |
| `coco` | COCO JSON format (pixel coordinates) | `annotations.json` (batch) or `*.json` (single) |
| `json` | Detailed JSON with confidence scores and preview | `*.json` + `*_preview.png` |

## CLI Options

| Flag | Description |
|------|-------------|
| `--format {yolo,coco,json}` | Output annotation format (default: yolo) |
| `--out OUT` | Output filename or directory |
| `--preview` | Generate preview image with bounding boxes (auto-enabled for JSON format) |
| `--min-confidence FLOAT` | Minimum confidence threshold 0-1 (default: 0.0) |
| `--adaptive` | Use Otsu's method for automatic thresholding |
| `--deskew` | Correct document skew before processing |
| `--min-area INT` | Minimum contour area in pixels (default: auto-scaled by DPI) |
| `--aggressive` | Maximum recall mode: relaxes filters, catches more but may have false positives |
| `--no-filter` | Disable all false positive filtering |

## Examples

```bash
# Standard detection on a single image
python detect_redactions.py document.png --format json --out result

# Batch process a folder with JSON output (includes previews)
python detect_redactions.py scanned_docs/ --format json --out detected/

# High-confidence detections only
python detect_redactions.py document.png --format json --min-confidence 0.8

# Handle skewed scans
python detect_redactions.py skewed_scan.png --format json --deskew

# Maximum recall (may include false positives, useful for pre-filtering)
python detect_redactions.py document.png --format json --aggressive
```

## How It Works

1. **Detection Pipeline**:
   - Converts image to grayscale and applies binary threshold
   - Uses morphological operations (CLOSE to fill gaps, OPEN to remove noise)
   - Finds contours and filters by area
   - Separately detects solid black regions to catch small inline redactions

2. **False Positive Filtering**:
   - Removes detections touching image edges (scanner artifacts)
   - Filters by aspect ratio (redactions are typically horizontal)
   - Requires solid black fill characteristics (low mean pixel value, low std deviation, high dark ratio)
   - Deduplicates overlapping detections

3. **Confidence Scoring**:
   - Based on dark pixel ratio, mean pixel value, aspect ratio, and edge distance
   - Preview images color-code by confidence: green (high), yellow (medium), red (low)

## Batch Output Structure

```
output_folder/
├── image1.json           # Detection data
├── image1_preview.png    # Annotated preview
├── image2.json
├── image2_preview.png
├── batch_summary.csv     # Quick summary
└── batch_summary.json    # Detailed summary
```

## Notes

- For manual inspection and labeling, consider [Label Studio](https://labelstud.io/guide/quick_start):
  ```bash
  pip install label-studio
  label-studio start
  ```
  Then open `http://localhost:8080`.
