import argparse
import base64
import os
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
import torch
from werkzeug.exceptions import RequestEntityTooLarge

from flask import Flask, jsonify, render_template, request
from torchvision import transforms
from ultralytics import YOLO

try:
    from fer import FER
except Exception:  # pragma: no cover
    FER = None

from src.models.caption_model import CaptionNet
from src.utils.text import Vocabulary
from src.utils.vision import enhance_frame, filter_small_boxes

app = Flask(__name__)


@app.errorhandler(RequestEntityTooLarge)
def _handle_upload_too_large(_e):
    return (
        jsonify(
            {
                "ok": False,
                "error": "Request body too large. Send a smaller JPEG or set MAX_UPLOAD_MB / --max-upload-mb.",
            }
        ),
        413,
    )


@app.after_request
def _cors(response):
    if app.config.get("ENABLE_CORS"):
        response.headers["Access-Control-Allow-Origin"] = app.config.get("CORS_ORIGIN", "*")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Max-Age"] = "86400"
    return response


WEARABLE_OBJECTS = {
    "backpack",
    "handbag",
    "tie",
    "suitcase",
    "umbrella",
    "hat",
    "shoe",
}


def frame_to_tensor(frame_bgr, device):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(rgb).unsqueeze(0).to(device)


@torch.no_grad()
def generate_caption(model, vocab, frame_bgr, max_len, device):
    image = frame_to_tensor(frame_bgr, device)
    features = model.encoder(image)
    ids = model.decoder.sample(
        features=features,
        max_len=max_len,
        start_id=vocab.stoi["<start>"],
        end_id=vocab.stoi["<end>"],
    )
    caption = vocab.decode(ids)
    return caption if caption else "unable to generate caption"


def load_caption_model(weights_path, vocab_path, device):
    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(weights_path, map_location=device)
    model = CaptionNet(
        embed_size=ckpt["embed_size"],
        hidden_size=ckpt["hidden_size"],
        vocab_size=ckpt["vocab_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab, ckpt.get("max_len", 25)


def movement_from_delta(dx, dy, threshold=12.0):
    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"
    horizontal = "right" if dx > threshold else "left" if dx < -threshold else ""
    vertical = "down" if dy > threshold else "up" if dy < -threshold else ""
    if horizontal and vertical:
        return f"moving {horizontal}-{vertical}"
    if horizontal:
        return f"moving {horizontal}"
    if vertical:
        return f"moving {vertical}"
    return "moving"


def summarize_detected_items(details):
    if not details:
        return "none"
    counts = {}
    for item in details:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1
    ordered = sorted(counts.items(), key=lambda x: x[0])
    return ", ".join(f"{count} {label}" for label, count in ordered)


def build_scene_description(base_caption, details):
    if not details:
        return f"{base_caption}. No clear objects are detected in this frame."
    detail_parts = [f"{item['label']} is {item['movement']}" for item in details]
    detail_sentence = "; ".join(detail_parts)
    return f"{base_caption}. In the surroundings: {detail_sentence}."


def estimate_distance(area_ratio):
    if area_ratio > 0.20:
        return "very close"
    if area_ratio > 0.10:
        return "close"
    if area_ratio > 0.04:
        return "medium distance"
    return "far"


def estimate_position(cx, cy, frame_w, frame_h):
    if cx < frame_w * 0.33:
        x_pos = "left"
    elif cx > frame_w * 0.66:
        x_pos = "right"
    else:
        x_pos = "center"
    if cy < frame_h * 0.33:
        y_pos = "upper"
    elif cy > frame_h * 0.66:
        y_pos = "lower"
    else:
        y_pos = "middle"
    return f"{y_pos}-{x_pos}"


def describe_surroundings_characteristics(frame, item_counts):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 70:
        light_text = "dark lighting"
    elif brightness < 150:
        light_text = "moderate lighting"
    else:
        light_text = "bright lighting"
    if not item_counts:
        return f"Scene has {light_text} and no clear objects."
    object_text = ", ".join(f"{count} {label}" for label, count in sorted(item_counts.items()))
    return f"Scene has {light_text}. Visible objects: {object_text}."


def color_name_from_bgr(mean_bgr):
    b, g, r = [int(v) for v in mean_bgr]
    if max(r, g, b) < 55:
        return "black"
    if min(r, g, b) > 200:
        return "white"
    if abs(r - g) < 20 and abs(g - b) < 20:
        return "gray"
    if r > g + 25 and r > b + 25:
        return "red"
    if g > r + 25 and g > b + 25:
        return "green"
    if b > r + 25 and b > g + 25:
        return "blue"
    if r > 140 and g > 110 and b < 110:
        return "yellow"
    if r > 130 and b > 130 and g < 110:
        return "purple"
    if r > 145 and g > 75 and b < 90:
        return "orange"
    return "mixed color"


def estimate_clothing_colors(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return "unknown top", "unknown bottom"
    person_roi = frame[y1:y2, x1:x2]
    if person_roi.size == 0:
        return "unknown top", "unknown bottom"
    roi_h = person_roi.shape[0]
    top_roi = person_roi[: max(1, roi_h // 2), :]
    bottom_roi = person_roi[max(1, roi_h // 2) :, :]
    top_color = color_name_from_bgr(top_roi.reshape(-1, 3).mean(axis=0))
    bottom_color = color_name_from_bgr(bottom_roi.reshape(-1, 3).mean(axis=0))
    return f"{top_color} top", f"{bottom_color} bottom"


def detect_faces_in_frame(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
    return faces


def build_face_environment_summary(face_count, item_counts):
    if item_counts:
        common = Counter(item_counts).most_common(3)
        top_items = ", ".join(f"{name}({count})" for name, count in common)
    else:
        top_items = "no clear objects"
    return f"Faces detected: {face_count}. Environment objects: {top_items}."


def collect_wearable_items(labels):
    wearables = [label for label in labels if label in WEARABLE_OBJECTS]
    if not wearables:
        return "no clearly detected accessories"
    counts = Counter(wearables)
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))


def build_nlp_report(base_caption, person_text, surroundings_text, emotion_text, wearables_text):
    return (
        f"{base_caption}. Person analysis: {person_text}. Clothing and accessories: {wearables_text}. "
        f"Scene summary: {surroundings_text}. Face emotion summary: {emotion_text}."
    )


def detect_emotions(frame, emotion_detector):
    if emotion_detector is None:
        return []
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = emotion_detector.detect_emotions(rgb)
    result = []
    for pred in predictions:
        emotions = pred.get("emotions", {})
        if not emotions:
            continue
        best = max(emotions, key=emotions.get)
        score = float(emotions.get(best, 0.0))
        result.append({"emotion": best, "score": score, "box": pred.get("box", None)})
    return result


def format_emotion_summary(emotions):
    if not emotions:
        return "No clear face emotion detected."
    phrases = []
    for idx, item in enumerate(emotions, start=1):
        confidence = int(item["score"] * 100)
        phrases.append(f"Face {idx}: {item['emotion']} ({confidence}% confidence)")
    return " | ".join(phrases)


class _ModelBundle:
    def __init__(self, caption_model, vocab, max_len, detector, device, face_cascade, emotion_detector):
        self.caption_model = caption_model
        self.vocab = vocab
        self.max_len = max_len
        self.detector = detector
        self.device = device
        self.face_cascade = face_cascade
        self.emotion_detector = emotion_detector


_bundle_lock = threading.Lock()
_bundle = None

_client_state_lock = threading.Lock()
_previous_centers_by_client = {}
_last_seen_by_client = {}
_last_caption_time_by_client = {}
_last_caption_by_client = {}
_history_by_client = {}

_keras_emotion_lock = threading.Lock()
_keras_emotion_model = None
_keras_emotion_load_failed = False


def _keras_emotion_labels():
    raw = app.config.get("KERAS_EMOTION_LABELS")
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _get_keras_emotion_model():
    """Lazy-load optional Keras HDF5 emotion classifier (e.g. emotion_model.hdf5)."""
    global _keras_emotion_model, _keras_emotion_load_failed
    path = (app.config.get("KERAS_EMOTION_MODEL") or "").strip()
    if not path:
        return None
    if _keras_emotion_load_failed:
        return None
    with _keras_emotion_lock:
        if _keras_emotion_model is None and not _keras_emotion_load_failed:
            try:
                from tensorflow.keras.models import load_model as keras_load_model  # noqa: WPS433

                if not os.path.exists(path):
                    _keras_emotion_load_failed = True
                    return None
                _keras_emotion_model = keras_load_model(path)
            except Exception:
                _keras_emotion_model = None
                _keras_emotion_load_failed = True
                return None
        return _keras_emotion_model


def keras_predict_emotion(frame_bgr, face_cascade):
    """
    Match classic FER-style pipeline: largest Haar face -> 48x48 gray -> model.
    Returns (label_or_status, confidence_or_None).
    """
    model = _get_keras_emotion_model()
    if model is None:
        return None, None
    faces = detect_faces_in_frame(frame_bgr, face_cascade)
    if len(faces) == 0:
        return "No face detected", None
    fx, fy, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    roi = frame_bgr[int(fy) : int(fy + fh), int(fx) : int(fx + fw)]
    if roi.size == 0:
        return None, None
    if roi.ndim == 2:
        gray = roi
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.astype(np.float32) / 255.0
    face_in = np.reshape(face, (1, 48, 48, 1))
    preds = model.predict(face_in, verbose=0)
    logits = np.asarray(preds[0]).flatten()
    idx = int(np.argmax(logits))
    labels = _keras_emotion_labels()
    if labels:
        idx = min(idx, len(labels) - 1)
        emotion = labels[idx]
    else:
        emotion = str(idx)
    conf = float(np.max(logits))
    return emotion, conf


def _frame_from_json_image_array(arr):
    """Build BGR frame from nested list (same idea as request.json['image'])."""
    img = np.array(arr, dtype=np.uint8)
    if img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _truthy(value):
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _smoothed(history, key, fallback=None):
    if not history:
        return fallback
    counts = Counter()
    for item in history:
        val = item.get(key)
        if not val:
            continue
        counts[str(val)] += 1
    if not counts:
        return fallback
    return counts.most_common(1)[0][0]


def _smoothed_detected(history):
    if not history:
        return "none"
    total = Counter()
    for item in history:
        counts = item.get("detected_counts") or {}
        total.update(counts)
    if not total:
        return "none"
    ordered = sorted(total.items(), key=lambda x: x[0])
    return ", ".join(f"{count} {label}" for label, count in ordered)


def _get_bundle():
    global _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = app.config["WEIGHTS"]
        vocab_path = app.config["VOCAB"]
        if not os.path.exists(weights) or not os.path.exists(vocab_path):
            raise FileNotFoundError(
                "Model artifacts not found. Ensure WEIGHTS and VOCAB exist (e.g. artifacts/caption_model.pt and artifacts/vocab.json)."
            )

        caption_model, vocab, max_len = load_caption_model(weights, vocab_path, device)
        detector = YOLO(app.config["DETECTOR"])
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        emotion_detector = FER(mtcnn=False) if FER is not None else None
        _bundle = _ModelBundle(
            caption_model=caption_model,
            vocab=vocab,
            max_len=max_len,
            detector=detector,
            device=device,
            face_cascade=face_cascade,
            emotion_detector=emotion_detector,
        )
        return _bundle


def _cleanup_client_state(now, ttl_seconds=30.0):
    with _client_state_lock:
        stale = [cid for cid, ts in _last_seen_by_client.items() if now - ts > ttl_seconds]
        for cid in stale:
            _previous_centers_by_client.pop(cid, None)
            _last_seen_by_client.pop(cid, None)
            _last_caption_time_by_client.pop(cid, None)
            _last_caption_by_client.pop(cid, None)
            _history_by_client.pop(cid, None)


def _analyze_frame(frame_bgr, client_id="default"):
    bundle = _get_bundle()
    now = time.time()
    _cleanup_client_state(now)

    frame_for_model = enhance_frame(frame_bgr)

    caption_interval = max(0.1, app.config["CAPTION_INTERVAL"])
    smooth_window = int(app.config.get("SMOOTH_WINDOW", 4))
    base_accuracy_mode = bool(app.config.get("ACCURACY_MODE", False))
    with _client_state_lock:
        last_caption_time = _last_caption_time_by_client.get(client_id, 0.0)
        cached_caption = _last_caption_by_client.get(client_id, "Initializing...")
        previous_centers = _previous_centers_by_client.get(client_id, {})
        history = _history_by_client.get(client_id)
        if history is None:
            history = deque(maxlen=max(1, smooth_window))
            _history_by_client[client_id] = history
        _last_seen_by_client[client_id] = now

    if now - last_caption_time >= caption_interval:
        caption = generate_caption(
            bundle.caption_model, bundle.vocab, frame_for_model, bundle.max_len, bundle.device
        )
        with _client_state_lock:
            _last_caption_time_by_client[client_id] = now
            _last_caption_by_client[client_id] = caption
        cached_caption = caption

    det_conf = app.config["DET_CONF"]
    det_iou = app.config["DET_IOU"]
    img_size = app.config["IMG_SIZE"]
    min_area_ratio = app.config["MIN_AREA_RATIO"]

    request_accuracy = None
    try:
        request_accuracy = request.args.get("accuracy")
    except Exception:
        request_accuracy = None
    accuracy_mode = _truthy(request_accuracy) if request_accuracy is not None else base_accuracy_mode

    if accuracy_mode:
        try:
            results = bundle.detector.track(
                frame_for_model,
                conf=det_conf,
                iou=det_iou,
                imgsz=img_size,
                persist=True,
                verbose=False,
                tracker=app.config.get("TRACKER", "bytetrack.yaml"),
            )
        except Exception:
            results = bundle.detector.predict(
                frame_for_model, conf=det_conf, iou=det_iou, imgsz=img_size, verbose=False
            )
    else:
        results = bundle.detector.predict(
            frame_for_model, conf=det_conf, iou=det_iou, imgsz=img_size, verbose=False
        )
    annotated = results[0].plot() if results else frame_bgr.copy()

    labels = []
    movement_details = []
    current_centers = {}
    person_details = []
    frame_h, frame_w = frame_bgr.shape[:2]
    if results and len(results[0].boxes) > 0:
        names = results[0].names
        cls_list = results[0].boxes.cls.tolist()
        xyxy_list = results[0].boxes.xyxy.tolist()
        id_list = None
        if getattr(results[0].boxes, "id", None) is not None:
            try:
                id_list = results[0].boxes.id.tolist()
            except Exception:
                id_list = None

        valid_indices = filter_small_boxes(results[0], min_area_ratio=min_area_ratio)
        for idx in valid_indices:
            cls_id = cls_list[idx]
            label = names[int(cls_id)]
            labels.append(label)
            x1, y1, x2, y2 = xyxy_list[idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            track_id = None
            if id_list is not None:
                try:
                    track_id = int(id_list[idx])
                except Exception:
                    track_id = None
            track_key = f"{label}_{track_id}" if track_id is not None else f"{label}_{idx}"
            current_centers[track_key] = (cx, cy)

            previous = previous_centers.get(track_key)
            if previous is None:
                movement = "newly appeared"
            else:
                movement = movement_from_delta(cx - previous[0], cy - previous[1])
            movement_details.append({"label": label, "movement": movement})

            if label == "person":
                box_area = max(1.0, (x2 - x1) * (y2 - y1))
                area_ratio = box_area / float(frame_w * frame_h)
                top_wear, bottom_wear = estimate_clothing_colors(frame_bgr, x1, y1, x2, y2)
                person_details.append(
                    {
                        "distance": estimate_distance(area_ratio),
                        "position": estimate_position(cx, cy, frame_w, frame_h),
                        "movement": movement,
                        "area_ratio": area_ratio,
                        "top_wear": top_wear,
                        "bottom_wear": bottom_wear,
                    }
                )

    with _client_state_lock:
        _previous_centers_by_client[client_id] = current_centers

    detected_text = summarize_detected_items(movement_details)
    movement_text = "; ".join(
        f"{item['label']}: {item['movement']}" for item in movement_details
    ) if movement_details else "none"
    item_counts = {}
    for item in movement_details:
        label = item["label"]
        item_counts[label] = item_counts.get(label, 0) + 1

    with _client_state_lock:
        history.append(
            {
                "detected_counts": dict(item_counts),
                "caption": cached_caption,
                "face_emotion": None,
                "keras_emotion": None,
            }
        )

    if person_details:
        person_lines = []
        for idx, p in enumerate(person_details, start=1):
            person_lines.append(
                f"Person {idx}: {p['distance']}, {p['position']}, {p['movement']}, "
                f"wearing {p['top_wear']} and {p['bottom_wear']}"
            )
        person_text = " | ".join(person_lines)
    else:
        person_text = "No person detected."

    surroundings_text = describe_surroundings_characteristics(frame_bgr, item_counts)
    wearables_text = collect_wearable_items(labels)

    faces = detect_faces_in_frame(frame_bgr, bundle.face_cascade)
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (255, 0, 255), 2)

    emotion_predictions = detect_emotions(frame_bgr, bundle.emotion_detector)
    for item in emotion_predictions:
        box = item.get("box")
        if not box:
            continue
        ex, ey, ew, eh = box
        cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), (0, 165, 255), 2)
        cv2.putText(
            annotated,
            item["emotion"],
            (ex, max(20, ey - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    face_environment_text = build_face_environment_summary(len(faces), item_counts)
    face_emotion_text = format_emotion_summary(emotion_predictions)

    keras_label, keras_conf = keras_predict_emotion(frame_bgr, bundle.face_cascade)
    if keras_label is None:
        keras_emotion_summary = "Keras: (model not configured)"
    elif keras_conf is None:
        keras_emotion_summary = f"Keras: {keras_label}"
    else:
        keras_emotion_summary = f"Keras: {keras_label} ({int(keras_conf * 100)}%)"

    nlp_report_text = build_nlp_report(
        cached_caption, person_text, surroundings_text, face_emotion_text, wearables_text
    )
    scene_description_text = build_scene_description(cached_caption, movement_details)

    if accuracy_mode:
        with _client_state_lock:
            if history:
                history[-1]["face_emotion"] = face_emotion_text
                history[-1]["keras_emotion"] = keras_emotion_summary
        detected_text = _smoothed_detected(history)
        cached_caption = _smoothed(history, "caption", fallback=cached_caption) or cached_caption
        face_emotion_text = _smoothed(history, "face_emotion", fallback=face_emotion_text) or face_emotion_text
        keras_emotion_summary = (
            _smoothed(history, "keras_emotion", fallback=keras_emotion_summary) or keras_emotion_summary
        )
        scene_description_text = build_scene_description(cached_caption, movement_details)
        nlp_report_text = build_nlp_report(
            cached_caption, person_text, surroundings_text, face_emotion_text, wearables_text
        )

    cv2.putText(
        annotated,
        f"Detected: {detected_text}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Caption: {cached_caption[:90]}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if movement_details:
        movement_overlay = f"Movement: {movement_details[0]['label']} {movement_details[0]['movement']}"
        cv2.putText(
            annotated,
            movement_overlay[:100],
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"Faces: {len(faces)}",
        (10, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Emotion: {face_emotion_text[:70]}",
        (10, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        keras_emotion_summary[:75],
        (10, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (180, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Clothes/Accessories: {wearables_text[:70]}",
        (10, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (50, 220, 220),
        2,
        cv2.LINE_AA,
    )

    payload = {
        "caption": cached_caption,
        "detected": detected_text,
        "movement": movement_text,
        "scene_description": scene_description_text,
        "person_characteristics": person_text,
        "surroundings_characteristics": surroundings_text,
        "face_environment": face_environment_text,
        "face_emotion": face_emotion_text,
        "keras_emotion": keras_emotion_summary,
        "keras_emotion_label": keras_label,
        "keras_emotion_confidence": keras_conf,
        "nlp_report": nlp_report_text,
        "accuracy_mode": bool(accuracy_mode),
    }
    return annotated, payload


@app.route("/")
def index():
    return render_template("browser_cam.html")


@app.route("/browser_cam")
def browser_cam():
    return render_template("browser_cam.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "ml-camera-pipeline",
            "docs": "/api",
            "keras_emotion_configured": bool((app.config.get("KERAS_EMOTION_MODEL") or "").strip()),
        }
    )


@app.route("/api")
def api_info():
    """JSON discovery for clients (browser, EC2, mobile)."""
    base = request.url_root.rstrip("/")
    return jsonify(
        {
            "service": "ml-camera-pipeline",
            "version": "1",
            "endpoints": {
                "ui": {"method": "GET", "path": "/", "description": "Browser webcam demo"},
                "health": {"method": "GET", "path": "/health"},
                "predict": {
                    "method": "POST",
                    "path": "/predict",
                    "body": "multipart field 'frame' (JPEG) or JSON {image: data URL}; query ?client_id= & ?accuracy=1",
                },
                "analyze_frame": {
                    "method": "POST",
                    "path": "/analyze_frame",
                    "body": "same as /predict",
                },
                "keras_emotion": {
                    "method": "POST",
                    "path": "/predict_keras_emotion",
                    "body": "JSON {image_b64} or {image: nested array}",
                },
            },
            "example_curl_predict": (
                f'curl -X POST -F frame=@photo.jpg "{base}/predict?client_id=demo&accuracy=1"'
            ),
        }
    )


@app.route("/analyze_frame", methods=["POST", "OPTIONS"])
def analyze_frame():
    if request.method == "OPTIONS":
        return "", 204

    client_id = request.args.get("client_id", "browser")

    frame = None
    upload = request.files.get("frame") or request.files.get("file") or request.files.get("image")
    if upload is not None and upload.filename not in (None, ""):
        raw = upload.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif upload is not None:
        # Some clients send a blob without a filename; still read the stream.
        raw = upload.read()
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        data = request.get_json(silent=True) or {}
        data_url = data.get("image")
        if isinstance(data_url, str) and data_url.startswith("data:image"):
            try:
                b64 = data_url.split(",", 1)[1]
                raw = base64.b64decode(b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                frame = None

    if frame is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "No frame provided. Send multipart field 'frame' or JSON {image: dataURL}.",
                }
            ),
            400,
        )

    try:
        annotated, payload = _analyze_frame(frame, client_id=client_id)
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    except Exception as e:
        app.logger.exception("analyze_frame failed")
        return jsonify({"ok": False, "error": str(e)}), 500

    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return jsonify({"ok": False, "error": "Failed to encode annotated frame."}), 500

    payload["ok"] = True
    payload["annotated_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify(payload)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Alias for `/analyze_frame` (frontend/backends often expect `/predict`)
    if request.method == "OPTIONS":
        return "", 204
    return analyze_frame()


@app.route("/predict_keras_emotion", methods=["POST", "OPTIONS"])
def predict_keras_emotion():
    """
    Standalone endpoint compatible with small JSON emotion APIs.

    Body (JSON), either:
      {"image": <nested uint8 array>}   # same idea as your sample (large payloads)
      {"image_b64": "<jpeg base64>"}    # preferred for browsers

    Response:
      {"emotion": "<label>", "confidence": 0.0-1.0}  # confidence omitted if N/A
    """
    if request.method == "OPTIONS":
        return "", 204
    model = _get_keras_emotion_model()
    if model is None:
        return (
            jsonify(
                {
                    "error": "Keras emotion model not configured. "
                    "Set env KERAS_EMOTION_MODEL or pass --keras-emotion-model path "
                    "(e.g. emotion_model.hdf5)."
                }
            ),
            503,
        )

    data = request.get_json(silent=True) or {}
    frame = None

    if isinstance(data.get("image"), list):
        frame = _frame_from_json_image_array(data["image"])
    elif isinstance(data.get("image_b64"), str):
        raw_b64 = data["image_b64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(raw_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            frame = None

    if frame is None:
        return (
            jsonify({"error": "Provide JSON {\"image\": nested array} or {\"image_b64\": \"...\"}."}),
            400,
        )

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    label, conf = keras_predict_emotion(frame, cascade)
    if label is None:
        return jsonify({"error": "Emotion inference failed."}), 500

    out = {"emotion": label}
    if conf is not None:
        out["confidence"] = conf
    return jsonify(out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.getenv("WEIGHTS", "artifacts/caption_model.pt"))
    parser.add_argument("--vocab", default=os.getenv("VOCAB", "artifacts/vocab.json"))
    parser.add_argument("--detector", default=os.getenv("DETECTOR", "yolov8n.pt"))
    parser.add_argument("--det-conf", type=float, default=float(os.getenv("DET_CONF", "0.35")))
    parser.add_argument("--det-iou", type=float, default=float(os.getenv("DET_IOU", "0.45")))
    parser.add_argument("--img-size", type=int, default=int(os.getenv("IMG_SIZE", "960")))
    parser.add_argument("--min-area-ratio", type=float, default=float(os.getenv("MIN_AREA_RATIO", "0.0015")))
    parser.add_argument("--caption-interval", type=float, default=float(os.getenv("CAPTION_INTERVAL", "1.0")))
    parser.add_argument("--accuracy-mode", action="store_true", default=_truthy(os.getenv("ACCURACY_MODE", "0")))
    parser.add_argument("--smooth-window", type=int, default=int(os.getenv("SMOOTH_WINDOW", "4")))
    parser.add_argument("--tracker", default=os.getenv("TRACKER", "bytetrack.yaml"))
    parser.add_argument("--keras-emotion-model", default=os.getenv("KERAS_EMOTION_MODEL", ""))
    parser.add_argument(
        "--keras-emotion-labels",
        default=os.getenv(
            "KERAS_EMOTION_LABELS",
            "Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral",
        ),
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "5000")))
    parser.add_argument(
        "--enable-cors",
        action="store_true",
        default=_truthy(os.getenv("ENABLE_CORS", "0")),
        help="Send Access-Control-Allow-* headers (needed for some cross-origin setups).",
    )
    parser.add_argument(
        "--cors-origin",
        default=os.getenv("CORS_ORIGIN", "*"),
        help='Access-Control-Allow-Origin value (default "*").',
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(os.getenv("MAX_UPLOAD_MB", "25")),
        help="Max request body size in MB (JPEG uploads / JSON).",
    )
    args, _unknown = parser.parse_known_args()
    return args


def configure_app(args):
    app.config["WEIGHTS"] = args.weights
    app.config["VOCAB"] = args.vocab
    app.config["DETECTOR"] = args.detector
    app.config["DET_CONF"] = args.det_conf
    app.config["DET_IOU"] = args.det_iou
    app.config["IMG_SIZE"] = args.img_size
    app.config["MIN_AREA_RATIO"] = args.min_area_ratio
    app.config["CAPTION_INTERVAL"] = args.caption_interval
    app.config["ACCURACY_MODE"] = bool(args.accuracy_mode)
    app.config["SMOOTH_WINDOW"] = max(1, int(args.smooth_window))
    app.config["TRACKER"] = args.tracker
    app.config["KERAS_EMOTION_MODEL"] = (args.keras_emotion_model or "").strip()
    app.config["KERAS_EMOTION_LABELS"] = (args.keras_emotion_labels or "").strip()
    app.config["ENABLE_CORS"] = bool(args.enable_cors)
    app.config["CORS_ORIGIN"] = (args.cors_origin or "*").strip() or "*"
    max_mb = max(1, int(getattr(args, "max_upload_mb", 25)))
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024


if __name__ == "__main__":
    args = parse_args()
    configure_app(args)
    app.run(host=args.host, port=args.port, debug=False)

