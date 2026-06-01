import os
import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from ultralytics import YOLO
from sklearn.cluster import DBSCAN
from tensorflow.keras.layers import Dense


# ============================================================
# KERAS COMPATIBILITY FIX
# ============================================================
# The feature model was saved with a newer Keras field:
# quantization_config.
# Some Hugging Face/Keras versions do not understand this field.
# This patch removes it automatically when loading Dense layers.

_original_dense_from_config = Dense.from_config


@classmethod
def patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)


Dense.from_config = patched_dense_from_config


# ============================================================
# 1. FILE PATHS
# ============================================================

YOLO_MODEL_PATH = "models/best.pt"
FEATURE_MODEL_PATH = "models/product_recognizer_features.keras"
INDEX_FILE = "models/product_indexV2.faiss"
METADATA_FILE = "models/gallery_metadataV2.pkl"

PLANOGRAM_PATHS = {
    "shelf_a": "static/planograms/shelf_a.jpg",
    "shelf_b": "static/planograms/shelf_b.jpg",
    "shelf_c": "static/planograms/shelf_c.jpg",
    "shelf_d": "static/planograms/shelf_d.jpg",
    "shelf_e": "static/planograms/shelf_e.png",
}

TARGET_SHAPE = (224, 224)


# ============================================================
# 2. GLOBAL MODEL VARIABLES
# ============================================================

_models_loaded = False
YOLOmodel = None
feature_extractor = None
faiss_index = None
gallery_metadata = None
gallery_paths = None
gallery_labels = None


def load_models():
    global _models_loaded
    global YOLOmodel, feature_extractor, faiss_index
    global gallery_metadata, gallery_paths, gallery_labels

    if _models_loaded:
        return

    required_files = [
        YOLO_MODEL_PATH,
        FEATURE_MODEL_PATH,
        INDEX_FILE,
        METADATA_FILE,
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required file: {file_path}. "
                "Check that all model files are inside the models/ folder."
            )

    YOLOmodel = YOLO(YOLO_MODEL_PATH)

    feature_extractor = tf.keras.models.load_model(
        FEATURE_MODEL_PATH,
        compile=False,
        safe_mode=False
    )

    faiss_index = faiss.read_index(INDEX_FILE)

    with open(METADATA_FILE, "rb") as f:
        gallery_metadata = pickle.load(f)

    gallery_paths = gallery_metadata["paths"]
    gallery_labels = gallery_metadata["labels"]

    _models_loaded = True


# ============================================================
# 3. IMAGE LOADING
# ============================================================

def load_image_rgb(image_path):
    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# ============================================================
# 4. PRODUCT RECOGNITION HELPERS
# ============================================================

def prep_crop_for_identification(crop_rgb, contrast_factor=1.0):
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


def identify_product_crop_with_vector(crop_rgb, contrast_factor=1.0):
    if crop_rgb is None or crop_rgb.size == 0:
        return "Unknown", 0.0, None, None

    processed_tensor = prep_crop_for_identification(
        crop_rgb,
        contrast_factor=contrast_factor
    )

    embedding_vector = feature_extractor.predict(processed_tensor, verbose=0)
    embedding_vector = embedding_vector.astype("float32")

    faiss.normalize_L2(embedding_vector)

    flat_vector = embedding_vector.flatten()

    distances, indices = faiss_index.search(embedding_vector, k=1)
    matched_idx = int(indices[0][0])

    if matched_idx == -1 or matched_idx >= len(gallery_labels):
        return "Unknown", 0.0, None, None

    predicted_sku = str(gallery_labels[matched_idx])
    similarity = float(distances[0][0])
    reference_img_path = gallery_paths[matched_idx]

    return predicted_sku, similarity, reference_img_path, flat_vector


def similarity_status(similarity):
    if similarity >= 0.85:
        return "accepted"
    elif similarity >= 0.70:
        return "review"
    else:
        return "uncertain"


# ============================================================
# 5. TEAMMATE PIPELINE WRAPPED AS FUNCTION
# ============================================================

def run_teammate_pipeline(image_rgb, image_name="uploaded_image"):
    image_rgb = image_rgb.astype("uint8")
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    temp_path = f"temp_{image_name}.jpg"
    cv2.imwrite(temp_path, image_bgr)

    results = YOLOmodel.predict(temp_path, conf=0.25, verbose=False)

    if len(results[0].boxes) == 0:
        return pd.DataFrame(), image_rgb, []

    xywhn = results[0].boxes.xywhn.cpu().numpy()
    xyxy = results[0].boxes.xyxy.cpu().numpy()

    df = pd.DataFrame(
        xywhn,
        columns=["x_center", "y_center", "bbox_width", "bbox_height"]
    )

    df["image_name"] = image_name
    df["class_id"] = results[0].boxes.cls.cpu().numpy().astype(int)

    clusters = DBSCAN(
        eps=0.04,
        min_samples=3
    ).fit_predict(df[["y_center"]].values)

    centers = (
        df.groupby(clusters)["y_center"]
        .mean()
        .drop(-1, errors="ignore")
        .sort_values()
        .tolist()
    )

    if len(centers) == 0:
        centers = [float(df["y_center"].mean())]

    df["assigned_row"] = df["y_center"].apply(
        lambda y: [abs(y - r) for r in centers].index(
            min([abs(y - r) for r in centers])
        ) + 1
    )

    df["xmin"] = xyxy[:, 0]
    df["ymin"] = xyxy[:, 1]
    df["xmax"] = xyxy[:, 2]
    df["ymax"] = xyxy[:, 3]

    df = df.sort_values(
        by=["assigned_row", "x_center"]
    ).reset_index(drop=True)

    img = results[0].orig_img.copy()
    rgb_source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    unique_skus = sorted(list(set(gallery_labels)))
    matched_skus = []
    matched_vectors = []
    matched_similarities = []

    for row_num, items in df.groupby("assigned_row"):
        for idx, (_, item) in enumerate(items.iterrows(), start=1):
            x1 = int(item["xmin"])
            y1 = int(item["ymin"])
            x2 = int(item["xmax"])
            y2 = int(item["ymax"])

            crop = rgb_source[y1:y2, x1:x2]

            sku, similarity, _, feature_vec = identify_product_crop_with_vector(crop)

            matched_skus.append(sku)
            matched_vectors.append(feature_vec)
            matched_similarities.append(similarity)

            label = f"{sku} (R{row_num}-P{idx})"

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    matrix_df = pd.DataFrame(index=df.index, columns=unique_skus)

    for idx, (sku, vec) in enumerate(zip(matched_skus, matched_vectors)):
        if sku in matrix_df.columns and vec is not None:
            matrix_df.at[idx, sku] = vec

    final_planogram_matrix_df = pd.concat([df, matrix_df], axis=1)

    final_planogram_matrix_df["predicted_sku"] = matched_skus
    final_planogram_matrix_df["similarity_score"] = matched_similarities

    annotated_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return final_planogram_matrix_df, annotated_rgb, centers


# ============================================================
# 6. TABLE HELPERS
# ============================================================

def build_product_table(df_current):
    if df_current.empty:
        return []

    product_rows = []

    df_current = df_current.sort_values(
        ["assigned_row", "x_center"]
    ).reset_index(drop=True)

    df_current["object_id_in_row"] = (
        df_current.groupby("assigned_row").cumcount() + 1
    )

    for _, row in df_current.iterrows():
        similarity = float(row.get("similarity_score", 0.0))
        status = similarity_status(similarity)

        product_rows.append({
            "row_id": int(row["assigned_row"]),
            "object_id_in_row": int(row["object_id_in_row"]),
            "predicted_sku_ean": str(row.get("predicted_sku", "Unknown")),
            "predicted_category_id": "N/A",
            "similarity_score": round(similarity, 4),
            "status": status,
            "bbox": f"{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}",
        })

    return product_rows


def build_row_summary_from_df(df_current):
    if df_current.empty:
        return []

    rows = []
    row_counts = df_current.groupby("assigned_row").size()

    for row_id, count in row_counts.items():
        rows.append({
            "row_id": int(row_id),
            "total_detected": int(count),
        })

    return rows


# ============================================================
# 7. COMPARISON LOGIC
# ============================================================

def get_sku_columns(df):
    metadata_cols = {
        "xmin", "ymin", "xmax", "ymax",
        "assigned_row", "x_center", "y_center",
        "bbox_width", "bbox_height",
        "image_name", "class_id",
        "predicted_sku", "similarity_score"
    }

    return [c for c in df.columns if c not in metadata_cols]


def compare_ideal_vs_current(df_ideal, df_current):
    comparison_rows = []

    if df_ideal.empty:
        return comparison_rows

    sku_cols = get_sku_columns(df_ideal)

    for sku in sku_cols:
        expected = df_ideal[sku].notna().sum()

        if sku in df_current.columns:
            actual = df_current[sku].notna().sum()
        else:
            actual = 0

        if expected > 0 or actual > 0:
            missing = max(0, expected - actual)

            if missing > 0:
                status = "Missing"
                action = f"Restock {missing} unit(s)"
            else:
                status = "OK"
                action = "No restock needed"

            comparison_rows.append({
                "type": "SKU comparison",
                "row_id": "-",
                "sku": str(sku),
                "expected_quantity": int(expected),
                "current_quantity": int(actual),
                "missing_quantity": int(missing),
                "status": status,
                "action": action,
            })

    ideal_row_counts = df_ideal.groupby("assigned_row").size()
    current_row_counts = df_current.groupby("assigned_row").size()

    all_rows = sorted(
        list(set(ideal_row_counts.index).union(set(current_row_counts.index)))
    )

    for row_id in all_rows:
        expected = int(ideal_row_counts.get(row_id, 0))
        actual = int(current_row_counts.get(row_id, 0))
        missing = max(0, expected - actual)

        if actual < expected:
            status = "Under-stocked"
            action = f"Restock {missing} item(s)"
        elif actual > expected:
            status = "Extra items"
            action = f"Review {actual - expected} extra item(s)"
        else:
            status = "OK"
            action = "No restock needed"

        comparison_rows.append({
            "type": "Row count comparison",
            "row_id": int(row_id),
            "sku": "-",
            "expected_quantity": expected,
            "current_quantity": actual,
            "missing_quantity": missing,
            "status": status,
            "action": action,
        })

    return comparison_rows


# ============================================================
# 8. MAIN FUNCTION CALLED BY app.py
# ============================================================

def analyze_shelf_image(current_image_rgb, shelf_id="shelf_a"):
    load_models()

    if current_image_rgb is None:
        return {
            "annotated_image": None,
            "summary_text": "No current shelf image uploaded.",
            "products": [],
            "row_summary": [],
            "planogram_comparison": [],
        }

    if shelf_id not in PLANOGRAM_PATHS:
        raise ValueError(f"Unknown shelf_id: {shelf_id}")

    planogram_path = PLANOGRAM_PATHS[shelf_id]

    if not os.path.exists(planogram_path):
        raise FileNotFoundError(
            f"Stored planogram image not found: {planogram_path}"
        )

    planogram_rgb = load_image_rgb(planogram_path)

    df_ideal, _, ideal_row_centers = run_teammate_pipeline(
        planogram_rgb,
        image_name=f"{shelf_id}_planogram"
    )

    df_current, current_annotated, current_row_centers = run_teammate_pipeline(
        current_image_rgb,
        image_name=f"{shelf_id}_current"
    )

    comparison = compare_ideal_vs_current(df_ideal, df_current)

    product_table = build_product_table(df_current)
    row_summary = build_row_summary_from_df(df_current)

    total_missing = sum(
        row["missing_quantity"]
        for row in comparison
        if row["missing_quantity"] > 0
    )

    summary_text = (
        f"Shelf selected: {shelf_id}\n"
        f"Stored planogram image: {planogram_path}\n"
        f"Planogram detected products: {len(df_ideal)}\n"
        f"Current detected products: {len(df_current)}\n"
        f"Planogram detected rows: {len(ideal_row_centers)}\n"
        f"Current detected rows: {len(current_row_centers)}\n"
        f"Total missing quantity estimate: {total_missing}\n\n"
        "Note: This keeps the teammate pipeline logic. "
        "The backend automatically runs the same pipeline twice: "
        "once on the stored full-shelf planogram image and once on the uploaded current shelf image."
    )

    return {
        "annotated_image": current_annotated,
        "summary_text": summary_text,
        "products": product_table,
        "row_summary": row_summary,
        "planogram_comparison": comparison,
    }