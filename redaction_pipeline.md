# Redaction Detection Pipeline — Technical Reference

## 1. Input & DPI Estimation

- Accepts PNG, JPG, JPEG, TIFF, TIF, BMP images (read via `cv2.imread` as BGR)
- Estimates DPI by assuming **US Letter page size (8.5" x 11")**:
  - `estimated_dpi = avg(img_width / 8.5, img_height / 11.0)`
- All kernel sizes and area thresholds scale relative to a **BASE_DPI of 300**
  - `scale = (estimated_dpi / 300)^2` for area; `estimated_dpi / 300` for linear dimensions

## 2. Preprocessing

- **Grayscale conversion:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`
- **Optional deskewing** (`--deskew` flag):
  - Canny edge detection with thresholds **(50, 150)**, aperture 3
  - Hough Line Transform: rho=1, theta=pi/180, threshold=**200**
  - Takes **median angle** of detected horizontal lines
  - Only corrects if skew > **0.5 degrees**
  - Rotates via bilinear interpolation with border replication

## 3. Thresholding

Two modes:

- **Standard (default):** Fixed threshold of **50**, binary inverse — `cv2.threshold(gray, 50, 255, THRESH_BINARY_INV)`
- **Adaptive** (`--adaptive`): Otsu's method — `cv2.threshold(gray, 0, 255, THRESH_BINARY_INV + THRESH_OTSU)`

## 4. Morphological Operations

All kernel sizes scale with DPI. Base values (at 300 DPI):

| Kernel | Base Size | Min |
|--------|-----------|-----|
| Horizontal close | 50 x 1 | 10 x 1 |
| Open (noise removal) | 3 x 9 | 2 x 3 |
| Square (aggressive only) | 15 x 15 | 5 x 5 |

**Normal mode:** Horizontal close, then opening.

**Aggressive mode:** Additionally applies square closing, then unions the two results via bitwise OR before opening. This catches isolated word-level redactions.

## 5. Contour Detection & Area Filtering

After morphology, contours are extracted and filtered by minimum area:

| Mode | Base min area | Floor |
|------|--------------|-------|
| Normal | 2000 px^2 | 300 px^2 |
| Aggressive | 500 px^2 | 200 px^2 |

## 6. Solid Black Region Detection (Parallel Track)

Runs independently to catch small inline redactions missed by morphology:

- Threshold: **30** (very dark pixels only)
- Each contour must satisfy all three:
  - Area > `min_area // 2`
  - Region mean pixel value < **30**
  - Dark ratio > **0.7** (where dark = pixel value < 50)

Results are unioned with the morphological detections.

## 7. Deduplication

`deduplicate_boxes()` with **IoU threshold = 0.5**. Sorts by area (largest first), keeping the larger box when two overlap significantly.

## 8. False Positive Filtering

Controlled by `--no-filter` flag. In order, checks:

1. **Edge artifacts:** Removes any box touching the image boundary.
2. **Minimum height:** `max(5, int(10 * dpi_scale))` pixels — filters thin underlines/separators.
3. **Color saturation:** Converts region to HSV; rejects if saturation > **40** (colored annotations, not black redactions).
4. **Dark pixel ratio:** Requires at least **5%** dark pixels (normal) or **2%** (aggressive), where dark = pixel < 50.
5. **Aspect ratio:** Normal mode requires w/h >= **2.0**; aggressive mode disables this. Skipped for confirmed solid black boxes (mean < 30 AND dark_ratio > 0.8).
6. **Strict mode** (normal only): Checks whether the region looks like solid black vs. text. Accepts if any of:
   - Mean < **50** (very dark)
   - Dark ratio > **0.6** AND mean < **120**
   - Mean < **100** AND std < **70** AND dark ratio > **0.4**

## 9. Confidence Scoring

Penalty-based system starting at **1.0**:

| Condition | Adjustment |
|-----------|-----------|
| dark_ratio < 0.1 | -0.3 |
| dark_ratio < 0.2 | -0.15 |
| dark_ratio > 0.5 | +0.1 |
| Aspect ratio outside [3, 50] (normal) or [0.5, 100] (aggressive) | -0.25 / -0.15 |
| Edge distance < 20px | -0.3 |

Clamped to [0.0, 1.0]. Optional `--min-confidence` flag filters output.

## 10. Classification

**Redaction type** (uses content width = image width x 0.8, i.e., 10% margin each side):

- `height_ratio > 1.5` — **block** (multi-line)
- `width_ratio > 0.70` — **full_line**
- `width_ratio > 0.15` — **partial_line**
- else — **inline**

**Vertical zone** (by y-center position):

- < 15% — **header**
- > 85% — **footer**
- else — **body**

## 11. Metrics & Intensity

**Typical height estimation:** Median of box heights after filtering outliers (> 10px and < 95th percentile). Line-equivalent count = `sum(ceil(h / typical_height))`.

**Coverage:** `total_redaction_area / content_area` where content area uses 10% margins on all sides, capped at 1.0.

**Intensity classification:**

- <= 2 detections AND < 1% coverage — **light**
- > 10 detections OR > 5% coverage — **heavy**
- Otherwise — **moderate**

**Heaviness score (0-100):**

- 40% from count: `min(count/20, 1.0) x 40`
- 40% from coverage: `min(coverage%/10, 1.0) x 40`
- 20% from block count: `min(blocks/5, 1.0) x 20`

## 12. Output Formats

- **JSON:** Full per-detection metadata (bbox, confidence, type, zone) plus per-image summary.
- **YOLO:** Normalized `class_id x_center y_center width height`.
- **COCO:** Standard COCO object detection format (category: "redaction").

## 13. Batch Processing

- Groups images by top-level subfolder.
- Distributes folders across workers using greedy largest-first allocation.
- Generates `batch_summary.csv` and `batch_summary.json` with aggregate stats.

## 14. Pipeline Flow

```
Image -> BGR -> Grayscale -> [Deskew] -> DPI Estimation
  -> Threshold (50 or Otsu)
  -> Morphology (horizontal close + [square close] + open)
  -> Contour extraction (area filter)
  | PARALLEL: Solid black detection (threshold=30, mean<30, dark>70%)
  -> Union -> Deduplicate (IoU>0.5)
  -> False positive filtering (edge/height/color/dark/aspect/strict)
  -> Confidence scoring -> Classification -> Metrics -> Output
```

## 15. Configuration Flags

| Flag | Default | Effect |
|------|---------|--------|
| `--adaptive` | False | Use Otsu's thresholding instead of fixed 50 |
| `--deskew` | False | Correct document skew via Hough lines |
| `--no-filter` | False | Disable false positive filtering |
| `--min-confidence` | 0.0 | Minimum confidence threshold for output |
| `--min-area` | None | Override auto-scaled min_area threshold |
| `--aggressive` | False | Maximum recall: disable aspect ratio filter, lower min_area, add square kernel |
| `--workers` | 1 | Parallel worker count (0 = auto: cpu_count - 1) |

## 16. Key Design Choices

- **Dual detection tracks** (morphological + solid black) ensure both large redaction bars and small inline redactions are caught.
- **DPI-adaptive scaling** handles variable scan resolutions without manual tuning.
- **Normal vs. aggressive mode** trades precision for recall — normal filters out text-like regions; aggressive catches everything dark.
- **Edge artifact removal** handles scanner border marks common in scanned legal documents.
