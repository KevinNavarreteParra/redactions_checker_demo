"""
Streamlit app for reviewing redaction detection results.
Run with: streamlit run review_app.py
"""

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Constants
BATCH_RESULTS_DIR = "batch_results"
LOW_CONFIDENCE_THRESHOLD = 0.85
COLORS = {
    "approved": (0, 255, 0),      # Green
    "rejected": (255, 0, 0),      # Red
    "pending": (255, 255, 0),     # Yellow
    "selected": (0, 100, 255),    # Blue
}


def load_batch_summary():
    """Load the batch summary JSON file."""
    summary_path = Path(BATCH_RESULTS_DIR) / "batch_summary.json"
    if not summary_path.exists():
        st.error(f"Batch summary not found at {summary_path}")
        st.stop()
    with open(summary_path) as f:
        return json.load(f)


def load_document_json(relative_path: str) -> dict:
    """Load the JSON file for a specific document."""
    # Convert image path to JSON path
    json_path = Path(BATCH_RESULTS_DIR) / relative_path.replace(".png", ".json")
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


def save_document_json(relative_path: str, data: dict):
    """Save the JSON file for a specific document."""
    json_path = Path(BATCH_RESULTS_DIR) / relative_path.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)


def get_min_confidence(doc_data: dict) -> float:
    """Get the minimum confidence score for a document."""
    if not doc_data or not doc_data.get("detections"):
        return 1.0
    return min(d["confidence"] for d in doc_data["detections"])


def has_low_confidence_detections(doc_data: dict) -> bool:
    """Check if document has any low confidence detections."""
    return get_min_confidence(doc_data) < LOW_CONFIDENCE_THRESHOLD


def get_review_progress(doc_data: dict) -> tuple:
    """Get review progress for a document: (reviewed, total, approved)."""
    if not doc_data or not doc_data.get("detections"):
        return (0, 0, 0)
    detections = doc_data["detections"]
    total = len(detections)
    reviewed = sum(1 for d in detections if d["status"] != "pending")
    approved = sum(1 for d in detections if d["status"] == "approved")
    return (reviewed, total, approved)


def draw_boxes_on_image(image: Image.Image, detections: list, selected_id: int = None) -> Image.Image:
    """Draw bounding boxes on the image with color coding."""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    for det in detections:
        x, y, w, h = det["bbox"]
        status = det["status"]
        det_id = det["id"]

        # Choose color based on status and selection
        if det_id == selected_id:
            color = COLORS["selected"]
            width = 4
        else:
            color = COLORS.get(status, COLORS["pending"])
            width = 2

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

        # Draw ID label
        label = f"#{det_id}"
        draw.text((x + 2, y + 2), label, fill=color)

    return img_copy


def filter_and_sort_documents(batch_summary: dict, filter_low_conf: bool, sort_by: str) -> list:
    """Filter and sort documents based on settings."""
    documents = []

    for result in batch_summary["results"]:
        if result["status"] != "success":
            continue

        rel_path = result["relative_path"]
        doc_data = load_document_json(rel_path)

        if doc_data is None:
            continue

        # Apply filter
        if filter_low_conf and not has_low_confidence_detections(doc_data):
            continue

        min_conf = get_min_confidence(doc_data)
        coverage = result.get("coverage_percent", 0)

        documents.append({
            "relative_path": rel_path,
            "min_confidence": min_conf,
            "coverage": coverage,
            "detections": result.get("detections", 0),
        })

    # Sort
    if sort_by == "Lowest confidence":
        documents.sort(key=lambda x: x["min_confidence"])
    elif sort_by == "Highest coverage":
        documents.sort(key=lambda x: -x["coverage"])
    elif sort_by == "Alphabetical":
        documents.sort(key=lambda x: x["relative_path"])

    return documents


def export_yolo(output_dir: str):
    """Export approved detections to YOLO format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_summary = load_batch_summary()
    exported_count = 0

    for result in batch_summary["results"]:
        if result["status"] != "success":
            continue

        rel_path = result["relative_path"]
        doc_data = load_document_json(rel_path)

        if doc_data is None:
            continue

        # Get approved detections
        approved = [d for d in doc_data["detections"] if d["status"] == "approved"]
        if not approved:
            continue

        # Get image dimensions for normalization
        img_w, img_h = doc_data["image_size"]

        # Create YOLO format output
        yolo_lines = []
        for det in approved:
            x, y, w, h = det["bbox"]
            # Convert to YOLO format (normalized center coords)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h
            # Class 0 = redaction
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")

        # Write to file
        yolo_file = output_path / rel_path.replace(".png", ".txt")
        yolo_file.parent.mkdir(parents=True, exist_ok=True)
        with open(yolo_file, "w") as f:
            f.write("\n".join(yolo_lines))

        exported_count += 1

    return exported_count


def export_coco(output_file: str):
    """Export approved detections to COCO format."""
    batch_summary = load_batch_summary()

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "redaction"}]
    }

    image_id = 0
    annotation_id = 0

    for result in batch_summary["results"]:
        if result["status"] != "success":
            continue

        rel_path = result["relative_path"]
        doc_data = load_document_json(rel_path)

        if doc_data is None:
            continue

        # Get approved detections
        approved = [d for d in doc_data["detections"] if d["status"] == "approved"]
        if not approved:
            continue

        img_w, img_h = doc_data["image_size"]

        # Add image entry
        coco["images"].append({
            "id": image_id,
            "file_name": rel_path,
            "width": img_w,
            "height": img_h
        })

        # Add annotations
        for det in approved:
            x, y, w, h = det["bbox"]
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 0,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    return len(coco["images"]), annotation_id


def main():
    st.set_page_config(
        page_title="Redaction Review",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Redaction Detection Review")

    # Load batch summary
    batch_summary = load_batch_summary()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        # Filter toggle
        filter_low_conf = st.toggle(
            "Show only pages with low-confidence detections",
            value=True,
            help=f"Hide pages where all detections have confidence >= {LOW_CONFIDENCE_THRESHOLD}"
        )

        # Sort dropdown
        sort_by = st.selectbox(
            "Sort by",
            ["Lowest confidence", "Highest coverage", "Alphabetical"],
            index=0
        )

        st.divider()

        # Get filtered/sorted document list
        documents = filter_and_sort_documents(batch_summary, filter_low_conf, sort_by)

        if not documents:
            st.warning("No documents match the current filter.")
            st.stop()

        # Document selector
        doc_options = [d["relative_path"] for d in documents]

        # Initialize session state for current document index
        if "current_doc_idx" not in st.session_state:
            st.session_state.current_doc_idx = 0

        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("< Prev", disabled=st.session_state.current_doc_idx == 0):
                st.session_state.current_doc_idx -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_doc_idx + 1} of {len(documents)}")
        with col3:
            if st.button("Next >", disabled=st.session_state.current_doc_idx >= len(documents) - 1):
                st.session_state.current_doc_idx += 1
                st.rerun()

        # Ensure index is valid
        st.session_state.current_doc_idx = min(st.session_state.current_doc_idx, len(documents) - 1)

        current_doc = documents[st.session_state.current_doc_idx]
        st.caption(f"Current: {current_doc['relative_path']}")

        st.divider()

        # Progress display
        st.subheader("Progress")
        total_reviewed = 0
        total_detections = 0
        total_approved = 0

        for doc in documents:
            doc_data = load_document_json(doc["relative_path"])
            if doc_data:
                r, t, a = get_review_progress(doc_data)
                total_reviewed += r
                total_detections += t
                total_approved += a

        st.metric("Detections Reviewed", f"{total_reviewed}/{total_detections}")
        st.metric("Approved", total_approved)

        st.divider()

        # Export section
        st.subheader("Export")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export YOLO"):
                count = export_yolo("exports/yolo")
                st.success(f"Exported {count} images to exports/yolo/")
        with col2:
            if st.button("Export COCO"):
                imgs, anns = export_coco("exports/coco/annotations.json")
                st.success(f"Exported {imgs} images, {anns} annotations")

    # Main content area
    rel_path = current_doc["relative_path"]
    doc_data = load_document_json(rel_path)

    if doc_data is None:
        st.error(f"Could not load document: {rel_path}")
        st.stop()

    # Load the original image
    image_path = doc_data["image_path"]
    if not os.path.exists(image_path):
        st.error(f"Image not found: {image_path}")
        st.stop()

    image = Image.open(image_path)

    # Initialize session state for selected detection
    if "selected_detection" not in st.session_state:
        st.session_state.selected_detection = None

    # Initialize drawing mode
    if "drawing_mode" not in st.session_state:
        st.session_state.drawing_mode = False

    # Two columns: image and detection list
    col_img, col_list = st.columns([3, 1])

    with col_list:
        st.subheader("Detections")

        # Drawing mode toggle
        st.session_state.drawing_mode = st.toggle("Add bounding box mode", value=st.session_state.drawing_mode)

        st.divider()

        # Bulk actions
        st.caption("Bulk Actions")
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Approve All Pending"):
                for det in doc_data["detections"]:
                    if det["status"] == "pending":
                        det["status"] = "approved"
                save_document_json(rel_path, doc_data)
                st.rerun()
        with bcol2:
            if st.button("Reject All Pending"):
                for det in doc_data["detections"]:
                    if det["status"] == "pending":
                        det["status"] = "rejected"
                save_document_json(rel_path, doc_data)
                st.rerun()

        st.divider()

        # Detection list
        for det in sorted(doc_data["detections"], key=lambda x: x["confidence"]):
            det_id = det["id"]
            conf = det["confidence"]
            status = det["status"]

            # Status indicator
            status_emoji = {"approved": "✅", "rejected": "❌", "pending": "⏳"}.get(status, "❓")

            # Confidence color
            if conf < 0.7:
                conf_color = "red"
            elif conf < LOW_CONFIDENCE_THRESHOLD:
                conf_color = "orange"
            else:
                conf_color = "green"

            with st.container():
                st.markdown(f"**#{det_id}** {status_emoji} - Conf: :{conf_color}[{conf:.2f}]")

                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    if st.button("✅", key=f"approve_{det_id}", disabled=status == "approved"):
                        det["status"] = "approved"
                        save_document_json(rel_path, doc_data)
                        st.rerun()
                with dcol2:
                    if st.button("❌", key=f"reject_{det_id}", disabled=status == "rejected"):
                        det["status"] = "rejected"
                        save_document_json(rel_path, doc_data)
                        st.rerun()

                st.divider()

    with col_img:
        if st.session_state.drawing_mode:
            # Use drawable canvas for adding new boxes
            st.caption("Draw a rectangle to add a new bounding box")

            # Scale image for canvas (max 800px wide)
            scale = min(800 / image.width, 1.0)
            canvas_width = int(image.width * scale)
            canvas_height = int(image.height * scale)

            # Draw existing boxes on background image
            bg_image = draw_boxes_on_image(image, doc_data["detections"])
            bg_image = bg_image.resize((canvas_width, canvas_height), Image.Resampling.LANCZOS)

            canvas_result = st_canvas(
                fill_color="rgba(0, 100, 255, 0.3)",
                stroke_width=2,
                stroke_color="#0064FF",
                background_image=bg_image,
                drawing_mode="rect",
                key="canvas",
                width=canvas_width,
                height=canvas_height,
            )

            # Process new rectangles
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                if objects:
                    # Get the last drawn rectangle
                    last_rect = objects[-1]
                    if last_rect.get("type") == "rect":
                        # Convert back to original image coordinates
                        x = int(last_rect["left"] / scale)
                        y = int(last_rect["top"] / scale)
                        w = int(last_rect["width"] / scale)
                        h = int(last_rect["height"] / scale)

                        # Check if this is a new box (not already saved)
                        existing_boxes = [(d["bbox"][0], d["bbox"][1]) for d in doc_data["detections"]]
                        if (x, y) not in existing_boxes and w > 10 and h > 10:
                            # Generate new ID
                            new_id = max([d["id"] for d in doc_data["detections"]], default=0) + 1

                            # Add new detection
                            new_detection = {
                                "id": new_id,
                                "bbox": [x, y, w, h],
                                "confidence": 1.0,
                                "confidence_factors": {"source": "manual"},
                                "status": "approved"
                            }
                            doc_data["detections"].append(new_detection)
                            save_document_json(rel_path, doc_data)
                            st.success(f"Added new bounding box #{new_id}")
                            st.rerun()
        else:
            # Display image with boxes (non-drawing mode)
            display_image = draw_boxes_on_image(
                image,
                doc_data["detections"],
                selected_id=st.session_state.selected_detection
            )
            st.image(display_image, use_container_width=True)

        # Document info
        st.caption(f"Image: {image_path} | Size: {image.width}x{image.height}")
        review_progress = get_review_progress(doc_data)
        st.caption(f"Reviewed: {review_progress[0]}/{review_progress[1]} | Approved: {review_progress[2]}")


if __name__ == "__main__":
    main()
