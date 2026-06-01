import base64
import cv2
import numpy as np

from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


PLANOGRAM_PATHS = {
    "shelf_a": "static/planograms/shelf_a.jpg",
    "shelf_b": "static/planograms/shelf_b.jpg",
    "shelf_c": "static/planograms/shelf_c.jpg",
    "shelf_d": "static/planograms/shelf_d.jpg",
    "shelf_e": "static/planograms/shelf_e.png",
}


def image_to_data_url(image_rgb):
    if image_rgb is None:
        return None

    image_bgr = cv2.cvtColor(image_rgb.astype("uint8"), cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".jpg", image_bgr)

    if not success:
        return None

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_frontend_planogram(comparison):
    """
    Converts backend comparison rows into the format expected by index.html.

    The pipeline currently returns:
    - SKU comparison rows with row_id = "-"
    - Row count comparison rows with real row_id values

    So for the HTML report, we use the row-count comparison rows.
    """
    expected_planogram = {}
    actual_planogram = {}

    for row in comparison:
        if row.get("type") != "Row count comparison":
            continue

        row_id = row.get("row_id")

        if row_id == "-" or row_id is None:
            continue

        row_key = f"row_{row_id}"

        expected_qty = int(row.get("expected_quantity", 0))
        current_qty = int(row.get("current_quantity", 0))

        expected_planogram[row_key] = {
            "Expected items": expected_qty
        }

        actual_planogram[row_key] = {
            "Detected items": current_qty
        }

    return expected_planogram, actual_planogram


def build_issue_lists(comparison):
    missing_items = []
    restock_items = []

    for row in comparison:
        if row.get("type") != "Row count comparison":
            continue

        missing_qty = int(row.get("missing_quantity", 0))

        if missing_qty > 0:
            row_id = row.get("row_id")
            item_text = f"Row {row_id} is under-stocked by {missing_qty} item(s)"
            missing_items.append(item_text)
            restock_items.append(item_text)

    return missing_items, restock_items


def calculate_compliance_score(comparison):
    total_expected = 0
    total_missing = 0

    for row in comparison:
        if row.get("type") != "Row count comparison":
            continue

        total_expected += int(row.get("expected_quantity", 0))
        total_missing += int(row.get("missing_quantity", 0))

    if total_expected == 0:
        return 100

    return round(100 * (1 - total_missing / total_expected))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    from pipeline import analyze_shelf_image

    if "shelf_image" not in request.files:
        return jsonify({"error": "No shelf image uploaded."}), 400

    shelf_id = request.form.get("shelf_id", "shelf_a")
    file = request.files["shelf_image"]

    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        return jsonify({"error": "Could not read uploaded image."}), 400

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    result = analyze_shelf_image(image_rgb, shelf_id)

    products = result.get("products", [])
    comparison = result.get("planogram_comparison", [])

    detected_rows = len(set([p["row_id"] for p in products])) if products else 0
    detected_products = len(products)

    expected_planogram, actual_planogram = build_frontend_planogram(comparison)
    missing_items, restock_items = build_issue_lists(comparison)
    compliance_score = calculate_compliance_score(comparison)

    total_restock_qty = 0
    for row in comparison:
        if row.get("type") == "Row count comparison":
            total_restock_qty += int(row.get("missing_quantity", 0))

    response = {
        "detected_rows": detected_rows,
        "detected_products": detected_products,
        "compliance_score": compliance_score,
        "restock_total": total_restock_qty,
        "annotated_image_url": image_to_data_url(result.get("annotated_image")),
        "planogram_image_url": PLANOGRAM_PATHS.get(shelf_id, ""),
        "products": products,
        "expected_planogram": expected_planogram,
        "actual_planogram": actual_planogram,
        "issues": {
            "missing": missing_items if missing_items else ["No missing products detected"],
            "misplaced": ["Not evaluated in this prototype"],
            "restock": restock_items if restock_items else ["No restock needed"],
        },
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)