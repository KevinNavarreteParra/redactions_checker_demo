"""
build_datasets.py — Generate page / document / arbitration-level CSV datasets
from batch redaction-detection output.

Usage:
    python build_datasets.py <batch_output_dir> [--lookup PATH] [--out PATH]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, mode, stdev


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lookup_table(path):
    """Read lookup CSV -> dict keyed by folder_name."""
    lookup = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[row["folder_name"]] = {
                "arbitration_id": row["arbitration_id"],
                "year": row["year"],
                "doc_name": row["doc_name"],
                "match_method": row["match_method"],
            }
    return lookup


def load_batch_summary(batch_dir):
    """Read batch_summary.json -> list of result dicts."""
    path = Path(batch_dir) / "batch_summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def load_page_json(json_path):
    """Read one per-image JSON -> flat dict of page-level fields."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w, img_h = data.get("image_size", [0, 0])
    processing = data.get("processing", {})
    coverage = data.get("coverage", {})
    classification = data.get("classification", {})
    spatial = data.get("spatial", {})
    intensity = data.get("intensity", {})
    summary = data.get("summary", {})

    return {
        "total_detections": summary.get("total", 0),
        "high_confidence": summary.get("high_confidence", 0),
        "medium_confidence": summary.get("medium_confidence", 0),
        "low_confidence": summary.get("low_confidence", 0),
        "coverage_fraction": coverage.get("coverage_fraction", 0.0),
        "coverage_percent": round(coverage.get("coverage_fraction", 0.0) * 100, 2),
        "redaction_area_px": coverage.get("redaction_area_px", 0),
        "content_area_px": coverage.get("content_area_px", 0),
        "inline_count": classification.get("inline_count", 0),
        "partial_line_count": classification.get("partial_line_count", 0),
        "full_line_count": classification.get("full_line_count", 0),
        "block_count": classification.get("block_count", 0),
        "header_count": spatial.get("header_count", 0),
        "body_count": spatial.get("body_count", 0),
        "footer_count": spatial.get("footer_count", 0),
        "vertical_spread": spatial.get("vertical_spread", 0.0),
        "redaction_level": intensity.get("redaction_level", "none"),
        "heaviness_score": intensity.get("heaviness_score", 0),
        "image_width": img_w,
        "image_height": img_h,
        "estimated_dpi": processing.get("estimated_dpi", ""),
        "skew_angle": processing.get("skew_angle", ""),
    }


# ---------------------------------------------------------------------------
# Page-level dataset
# ---------------------------------------------------------------------------

PAGE_FIELDS = [
    "relative_path", "folder_name", "page_filename",
    "arbitration_id", "year", "doc_name", "match_method",
    "status", "error",
    "total_detections", "high_confidence", "medium_confidence", "low_confidence",
    "coverage_fraction", "coverage_percent", "redaction_area_px", "content_area_px",
    "inline_count", "partial_line_count", "full_line_count", "block_count",
    "header_count", "body_count", "footer_count", "vertical_spread",
    "redaction_level", "heaviness_score",
    "image_width", "image_height", "estimated_dpi", "skew_angle",
]


def _empty_detection_fields():
    """Return zeroed detection fields for non-success pages."""
    return {
        "total_detections": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "coverage_fraction": 0.0,
        "coverage_percent": 0.0,
        "redaction_area_px": 0,
        "content_area_px": 0,
        "inline_count": 0,
        "partial_line_count": 0,
        "full_line_count": 0,
        "block_count": 0,
        "header_count": 0,
        "body_count": 0,
        "footer_count": 0,
        "vertical_spread": 0.0,
        "redaction_level": "none",
        "heaviness_score": 0,
        "image_width": 0,
        "image_height": 0,
        "estimated_dpi": "",
        "skew_angle": "",
    }


def build_page_rows(batch_dir, batch_results, lookup):
    """Build page-level rows from per-image JSONs and batch_summary fallback."""
    batch_dir = Path(batch_dir)
    rows = []
    unmatched_folders = set()
    has_root_images = False

    for result in batch_results:
        rel_path = result["relative_path"]
        rel = Path(rel_path)
        parts = rel.parts
        folder_name = parts[0] if len(parts) > 1 else "."
        page_filename = rel.name

        if folder_name == ".":
            has_root_images = True

        # Lookup join
        lk = lookup.get(folder_name, {})
        if not lk and folder_name != ".":
            unmatched_folders.add(folder_name)

        row = {
            "relative_path": rel_path,
            "folder_name": folder_name,
            "page_filename": page_filename,
            "arbitration_id": lk.get("arbitration_id", ""),
            "year": lk.get("year", ""),
            "doc_name": lk.get("doc_name", ""),
            "match_method": lk.get("match_method", ""),
            "status": result["status"],
            "error": result.get("error") or "",
        }

        if result["status"] == "success":
            # Try to load per-image JSON
            # JSON path mirrors the relative_path with .json extension
            json_stem = rel.with_suffix(".json")
            json_path = batch_dir / json_stem
            if json_path.exists():
                row.update(load_page_json(json_path))
            else:
                # Fallback to batch summary fields
                row.update(_empty_detection_fields())
                row["total_detections"] = result.get("detections", 0)
                row["coverage_percent"] = result.get("coverage_percent", 0.0)
                row["coverage_fraction"] = round(row["coverage_percent"] / 100, 4) if row["coverage_percent"] else 0.0
                row["redaction_level"] = result.get("redaction_level", "none")
                row["heaviness_score"] = result.get("heaviness_score", 0)
        else:
            row.update(_empty_detection_fields())
            # For non-success, pull what we can from batch_summary
            row["total_detections"] = result.get("detections", 0)
            row["coverage_percent"] = result.get("coverage_percent", 0.0)
            row["coverage_fraction"] = round(row["coverage_percent"] / 100, 4) if row["coverage_percent"] else 0.0
            row["redaction_level"] = result.get("redaction_level", "none")
            row["heaviness_score"] = result.get("heaviness_score", 0)

        rows.append(row)

    if has_root_images:
        print("Warning: Some images are directly in the batch root (no subfolder) — they won't match the lookup table.")
    if unmatched_folders:
        print(f"Warning: {len(unmatched_folders)} folder(s) not found in lookup table.")

    # Sort by (arbitration_id, folder_name, page_filename)
    rows.sort(key=lambda r: (r["arbitration_id"], r["folder_name"], r["page_filename"]))
    return rows


# ---------------------------------------------------------------------------
# Document-level aggregation
# ---------------------------------------------------------------------------

DOCUMENT_FIELDS = [
    "folder_name", "arbitration_id", "year", "doc_name", "match_method",
    "total_pages", "pages_with_detections", "pages_no_detections", "pages_with_errors",
    "total_detections",
    "mean_detections_per_page", "median_detections_per_page",
    "max_detections_on_page", "std_detections_per_page",
    "mean_coverage_percent", "median_coverage_percent",
    "max_coverage_percent", "std_coverage_percent",
    "total_redaction_area_px",
    "total_inline", "total_partial_line", "total_full_line", "total_block",
    "total_header", "total_body", "total_footer",
    "total_high_confidence", "total_medium_confidence", "total_low_confidence",
    "mean_heaviness_score", "max_heaviness_score", "modal_redaction_level",
    "has_any_redaction", "fraction_pages_with_redactions",
]


def _safe_stdev(values):
    """Return stdev or 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def _safe_mode(values):
    """Return mode with fallback for ties (Python 3.8+ mode raises on ties in <3.8)."""
    if not values:
        return "none"
    try:
        return mode(values)
    except Exception:
        # Fallback: count manually
        counts = defaultdict(int)
        for v in values:
            counts[v] += 1
        return max(counts, key=counts.get)


def aggregate_document_level(page_rows):
    """Group page rows by folder_name, compute aggregates."""
    groups = defaultdict(list)
    for row in page_rows:
        groups[row["folder_name"]].append(row)

    doc_rows = []
    for folder_name, pages in groups.items():
        first = pages[0]
        total_pages = len(pages)

        pages_with_detections = sum(
            1 for p in pages if p["status"] == "success" and p["total_detections"] > 0
        )
        pages_no_detections = sum(
            1 for p in pages if p["status"] in ("no_detections", "all_filtered", "confidence_filtered")
        )
        pages_with_errors = sum(1 for p in pages if p["status"] == "error")

        det_counts = [p["total_detections"] for p in pages]
        cov_pcts = [p["coverage_percent"] for p in pages]

        doc = {
            "folder_name": folder_name,
            "arbitration_id": first["arbitration_id"],
            "year": first["year"],
            "doc_name": first["doc_name"],
            "match_method": first["match_method"],
            "total_pages": total_pages,
            "pages_with_detections": pages_with_detections,
            "pages_no_detections": pages_no_detections,
            "pages_with_errors": pages_with_errors,
            "total_detections": sum(det_counts),
            "mean_detections_per_page": round(mean(det_counts), 2) if det_counts else 0,
            "median_detections_per_page": round(median(det_counts), 2) if det_counts else 0,
            "max_detections_on_page": max(det_counts) if det_counts else 0,
            "std_detections_per_page": round(_safe_stdev(det_counts), 2),
            "mean_coverage_percent": round(mean(cov_pcts), 2) if cov_pcts else 0,
            "median_coverage_percent": round(median(cov_pcts), 2) if cov_pcts else 0,
            "max_coverage_percent": round(max(cov_pcts), 2) if cov_pcts else 0,
            "std_coverage_percent": round(_safe_stdev(cov_pcts), 2),
            "total_redaction_area_px": sum(p["redaction_area_px"] for p in pages),
            "total_inline": sum(p["inline_count"] for p in pages),
            "total_partial_line": sum(p["partial_line_count"] for p in pages),
            "total_full_line": sum(p["full_line_count"] for p in pages),
            "total_block": sum(p["block_count"] for p in pages),
            "total_header": sum(p["header_count"] for p in pages),
            "total_body": sum(p["body_count"] for p in pages),
            "total_footer": sum(p["footer_count"] for p in pages),
            "total_high_confidence": sum(p["high_confidence"] for p in pages),
            "total_medium_confidence": sum(p["medium_confidence"] for p in pages),
            "total_low_confidence": sum(p["low_confidence"] for p in pages),
            "mean_heaviness_score": round(mean(p["heaviness_score"] for p in pages), 2),
            "max_heaviness_score": max(p["heaviness_score"] for p in pages),
            "modal_redaction_level": _safe_mode([p["redaction_level"] for p in pages]),
            "has_any_redaction": pages_with_detections > 0,
            "fraction_pages_with_redactions": round(pages_with_detections / total_pages, 4) if total_pages else 0,
        }
        doc_rows.append(doc)

    doc_rows.sort(key=lambda r: (r["arbitration_id"], r["folder_name"]))
    return doc_rows


# ---------------------------------------------------------------------------
# Arbitration-level aggregation
# ---------------------------------------------------------------------------

ARBITRATION_FIELDS = [
    "arbitration_id", "year",
    "total_documents", "documents_with_any_redaction", "fraction_documents_with_redactions",
    "total_pages", "pages_with_detections", "pages_with_errors",
    "fraction_pages_with_redactions",
    "total_detections",
    "mean_detections_per_document", "mean_detections_per_page",
    "max_detections_in_document",
    "mean_coverage_percent", "max_coverage_percent",
    "total_inline", "total_partial_line", "total_full_line", "total_block",
    "total_header", "total_body", "total_footer",
    "mean_heaviness_score", "max_heaviness_score",
]


def aggregate_arbitration_level(doc_rows, page_rows):
    """Group document rows by arbitration_id, compute aggregates.

    page_rows is needed for per-page statistics across all pages in an arbitration.
    """
    doc_groups = defaultdict(list)
    for doc in doc_rows:
        doc_groups[doc["arbitration_id"]].append(doc)

    page_groups = defaultdict(list)
    for page in page_rows:
        page_groups[page["arbitration_id"]].append(page)

    arb_rows = []
    for arb_id, docs in doc_groups.items():
        first = docs[0]
        pages = page_groups.get(arb_id, [])

        total_documents = len(docs)
        documents_with_any_redaction = sum(1 for d in docs if d["has_any_redaction"])
        total_pages = sum(d["total_pages"] for d in docs)
        pages_with_detections = sum(d["pages_with_detections"] for d in docs)
        pages_with_errors = sum(d["pages_with_errors"] for d in docs)
        total_detections = sum(d["total_detections"] for d in docs)

        # Per-page coverage across all pages in this arbitration
        page_cov_pcts = [p["coverage_percent"] for p in pages]
        page_heaviness = [p["heaviness_score"] for p in pages]

        arb = {
            "arbitration_id": arb_id,
            "year": first["year"],
            "total_documents": total_documents,
            "documents_with_any_redaction": documents_with_any_redaction,
            "fraction_documents_with_redactions": round(documents_with_any_redaction / total_documents, 4) if total_documents else 0,
            "total_pages": total_pages,
            "pages_with_detections": pages_with_detections,
            "pages_with_errors": pages_with_errors,
            "fraction_pages_with_redactions": round(pages_with_detections / total_pages, 4) if total_pages else 0,
            "total_detections": total_detections,
            "mean_detections_per_document": round(mean(d["total_detections"] for d in docs), 2) if docs else 0,
            "mean_detections_per_page": round(total_detections / total_pages, 2) if total_pages else 0,
            "max_detections_in_document": max(d["total_detections"] for d in docs) if docs else 0,
            "mean_coverage_percent": round(mean(page_cov_pcts), 2) if page_cov_pcts else 0,
            "max_coverage_percent": round(max(page_cov_pcts), 2) if page_cov_pcts else 0,
            "total_inline": sum(d["total_inline"] for d in docs),
            "total_partial_line": sum(d["total_partial_line"] for d in docs),
            "total_full_line": sum(d["total_full_line"] for d in docs),
            "total_block": sum(d["total_block"] for d in docs),
            "total_header": sum(d["total_header"] for d in docs),
            "total_body": sum(d["total_body"] for d in docs),
            "total_footer": sum(d["total_footer"] for d in docs),
            "mean_heaviness_score": round(mean(page_heaviness), 2) if page_heaviness else 0,
            "max_heaviness_score": max(page_heaviness) if page_heaviness else 0,
        }
        arb_rows.append(arb)

    arb_rows.sort(key=lambda r: (r["year"], r["arbitration_id"]))
    return arb_rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(rows, fieldnames, path):
    """Write list of dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Wrote {len(rows)} rows -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate page / document / arbitration-level CSV datasets from batch redaction output."
    )
    parser.add_argument(
        "batch_output_dir",
        help="Path to the batch output directory (from detect_redactions.py --format json)",
    )
    parser.add_argument(
        "--lookup",
        default="data/lookup_table.csv",
        help="Path to the folder->arbitration lookup table (default: data/lookup_table.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for the three CSVs (default: <batch_output_dir>/datasets)",
    )
    args = parser.parse_args()

    batch_dir = Path(args.batch_output_dir)
    lookup_path = Path(args.lookup)
    out_dir = Path(args.out) if args.out else batch_dir / "datasets"

    # Validate inputs
    if not batch_dir.is_dir():
        print(f"Error: batch output directory not found: {batch_dir}")
        sys.exit(1)
    if not (batch_dir / "batch_summary.json").exists():
        print(f"Error: batch_summary.json not found in {batch_dir}")
        sys.exit(1)
    if not lookup_path.exists():
        print(f"Error: lookup table not found: {lookup_path}")
        sys.exit(1)

    # Load data
    print(f"Loading lookup table from {lookup_path} ...")
    lookup = load_lookup_table(lookup_path)
    print(f"  {len(lookup)} entries loaded.")

    print(f"Loading batch summary from {batch_dir / 'batch_summary.json'} ...")
    batch_results = load_batch_summary(batch_dir)
    print(f"  {len(batch_results)} image results found.")

    # Build page-level
    print("Building page-level dataset ...")
    page_rows = build_page_rows(batch_dir, batch_results, lookup)
    print(f"  {len(page_rows)} page rows built.")

    # Build document-level
    print("Building document-level dataset ...")
    doc_rows = aggregate_document_level(page_rows)
    print(f"  {len(doc_rows)} document rows built.")

    # Build arbitration-level
    print("Building arbitration-level dataset ...")
    arb_rows = aggregate_arbitration_level(doc_rows, page_rows)
    print(f"  {len(arb_rows)} arbitration rows built.")

    # Write CSVs
    print(f"\nWriting CSVs to {out_dir} ...")
    write_csv(page_rows, PAGE_FIELDS, out_dir / "page_level.csv")
    write_csv(doc_rows, DOCUMENT_FIELDS, out_dir / "document_level.csv")
    write_csv(arb_rows, ARBITRATION_FIELDS, out_dir / "arbitration_level.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
