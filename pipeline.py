import os
import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ultralytics import YOLO
from sklearn.cluster import DBSCAN


# -----------------------------
# File paths
# -----------------------------
YOLO_MODEL_PATH = "models/best.pt"
FEATURE_MODEL_PATH = "models/product_recognizer_features.keras"
INDEX_FILE = "models/product_indexV2.faiss"
METADATA_FILE = "models/gallery_metadataV2.pkl"

TARGET_SHAPE = (224, 224)


# -----------------------------
# Lazy-loaded global models
# -----------------------------
_models_loaded = False
YOLOmodel = None
feature_extractor = None
faiss_index = None
gallery_metadata = None
gallery_paths = None
gallery_labels = None


def load_models():
    """
    Load all models once when the first user runs the app.
    """
    global _models_loaded
    global YOLOmodel, feature_extractor, faiss_index
    global gallery_metadata, gallery_paths, gallery_labels

    if _models_loaded:
        return

    required_files = [
        YOLO_MODEL_PATH,
        FEATURE_MODEL_PATH,
        INDEX_FILE,
        METADATA_FILE
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_path}. "
                "Please check that all model files are inside the models/ folder."
            )

    YOLOmodel = YOLO(YOLO_MODEL_PATH)

    feature_extractor = tf.keras.models.load_model(
        FEATURE_MODEL_PATH,
        compile=False
    )

    faiss_index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        gallery_metadata = pickle.load(f)

    if isinstance(gallery_metadata, dict):
        gallery_paths = gallery_metadata.get("paths", [])
        gallery_labels = gallery_metadata.get("labels", [])
    else:
        raise ValueError(
            "gallery_metadataV2.pkl must contain a dictionary with keys 'paths' and 'labels'."
        )

    if len(gallery_labels) == 0:
        raise ValueError("No gallery labels found in gallery_metadataV2.pkl.")

    _models_loaded = True


def prep_crop_for_identification(crop_rgb, contrast_factor=1.0):
    """
    Preprocess a product crop the same way as the Colab pipeline.
    """
    img_tensor = tf.convert_to_tensor(crop_rgb, dtype=tf.float32)

    if tf.reduce_max(img_tensor) > 1.0:
        img_tensor = img_tensor / 255.0

    img_tensor = tf.image.resize(img_tensor, TARGET_SHAPE)

    if contrast_factor != 1.0:
        img_tensor = tf.image.adjust_contrast(
            img_tensor,
            contrast_factor=contrast_factor
        )

    img_tensor = tf.keras.applications.mobilenet_v3.preprocess_input(img_tensor)

    return tf.expand_dims(img_tensor, axis=0)


def identify_product_crop(crop_rgb, contrast_factor=1.0):
    """
    Match one crop to the closest SKU using MobileNetV3 embeddings + FAISS.

    Returns:
        predicted_sku, similarity_score, reference_img_path
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return "Unknown", 0.0, None

    processed_tensor = prep_crop_for_identification(
        crop_rgb,
        contrast_factor=contrast_factor
    )

    embedding_vector = feature_extractor.predict(processed_tensor, verbose=0)
    embedding_vector = embedding_vector.astype("float32")

    faiss.normalize_L2(embedding_vector)

    distances, indices = faiss_index.search(embedding_vector, k=1)

    matched_idx = int(indices[0][0])

    if matched_idx == -1 or matched_idx >= len(gallery_labels):
        return "Unknown", 0.0, None

    predicted_sku = str(gallery_labels[matched_idx])
    similarity_score = float(distances[0][0])

    reference_img_path = None
    if gallery_paths and matched_idx < len(gallery_paths):
        reference_img_path = gallery_paths[matched_idx]

    return predicted_sku, similarity_score, reference_img_path


def similarity_status(similarity):
    """
    Converts FAISS similarity into a readable status.
    These thresholds can be tuned after testing.
    """
    if similarity >= 0.85:
        return "accepted"
    elif similarity >= 0.70:
        return "review"
    else:
        return "uncertain"


def assign_rows_from_yolo(df, eps=0.04, min_samples=3):
    """
    Assign shelf rows using DBSCAN over normalized y_center values.
    """
    if df.empty:
        return df, []

    clusters = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit_predict(df[["y_center"]].values)

    df["row_cluster"] = clusters

    valid_rows = df[df["row_cluster"] != -1].copy()

    if valid_rows.empty:
        df["assigned_row"] = 1
        return df, [float(df["y_center"].mean())]

    row_centers = (
        valid_rows.groupby("row_cluster")["y_center"]
        .mean()
        .sort_values()
        .tolist()
    )

    def nearest_row(y):
        distances = [abs(y - center) for center in row_centers]
        return distances.index(min(distances)) + 1

    df["assigned_row"] = df["y_center"].apply(nearest_row)

    return df, row_centers


def build_row_summary(products):
    """
    Build row-level summary for the Gradio table.
    """
    if len(products) == 0:
        return []

    product_df = pd.DataFrame(products)

    summary = (
        product_df
        .groupby("row_id")
        .agg(
            total_detected=("object_id_in_row", "count"),
            accepted_count=("status", lambda x: (x == "accepted").sum()),
            review_count=("status", lambda x: (x == "review").sum()),
            uncertain_count=("status", lambda x: (x == "uncertain").sum())
        )
        .reset_index()
    )

    return summary.to_dict(orient="records")


def analyze_shelf_image(image_rgb, shelf_id="Shelf A"):
    """
    Main function called by app.py.

    Input:
        image_rgb: Numpy RGB image from Gradio
        shelf_id: selected shelf name

    Output:
        dictionary containing:
        - annotated_image
        - summary_text
        - products
        - row_summary
        - planogram_comparison
    """
    load_models()

    if image_rgb is None:
        return {
            "annotated_image": None,
            "summary_text": "No image uploaded.",
            "products": [],
            "row_summary": [],
            "planogram_comparison": []
        }

    image_rgb = image_rgb.astype("uint8")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # YOLO works reliably with image paths, so save temporary input.
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, image_bgr)

    results = YOLOmodel.predict(temp_path, conf=0.25, verbose=False)

    if len(results[0].boxes) == 0:
        return {
            "annotated_image": image_rgb,
            "summary_text": f"Shelf: {shelf_id}\nNo products detected.",
            "products": [],
            "row_summary": [],
            "planogram_comparison": []
        }

    xywhn = results[0].boxes.xywhn.cpu().numpy()
    xyxy = results[0].boxes.xyxy.cpu().numpy()
    det_conf = results[0].boxes.conf.cpu().numpy()

    df = pd.DataFrame(
        xywhn,
        columns=["x_center", "y_center", "bbox_width", "bbox_height"]
    )

    df["det_confidence"] = det_conf

    df, row_centers = assign_rows_from_yolo(
        df,
        eps=0.04,
        min_samples=3
    )

    df["xmin"] = xyxy[:, 0].astype(int)
    df["ymin"] = xyxy[:, 1].astype(int)
    df["xmax"] = xyxy[:, 2].astype(int)
    df["ymax"] = xyxy[:, 3].astype(int)

    df = df.sort_values(["assigned_row", "x_center"]).reset_index(drop=True)
    df["object_id_in_row"] = df.groupby("assigned_row").cumcount() + 1

    annotated_image = image_rgb.copy()
    products = []

    for _, item in df.iterrows():
        row_id = int(item["assigned_row"])
        object_id = int(item["object_id_in_row"])

        x1 = int(item["xmin"])
        y1 = int(item["ymin"])
        x2 = int(item["xmax"])
        y2 = int(item["ymax"])

        crop = image_rgb[y1:y2, x1:x2]

        predicted_sku, similarity, _ = identify_product_crop(crop)
        status = similarity_status(similarity)

        label = f"{predicted_sku} (R{row_id}-P{object_id})"

        cv2.rectangle(
            annotated_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_image,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        products.append({
            "row_id": row_id,
            "object_id_in_row": object_id,
            "predicted_sku_ean": predicted_sku,
            "predicted_category_id": "N/A",
            "similarity_score": round(similarity, 4),
            "status": status,
            "bbox": f"{x1},{y1},{x2},{y2}",
            "det_confidence": round(float(item["det_confidence"]), 4)
        })

    row_summary = build_row_summary(products)

    summary_text = (
        f"Shelf: {shelf_id}\n"
        f"Detected rows: {len(row_centers)}\n"
        f"Detected products: {len(products)}\n"
        "Product detection, row assignment, and FAISS SKU retrieval completed.\n"
        "Planogram comparison will be connected once the final planogram CSV/logic is ready."
    )

    return {
        "annotated_image": annotated_image,
        "summary_text": summary_text,
        "products": products,
        "row_summary": row_summary,
        "planogram_comparison": []
    }