import argparse
import base64
import os
import sys
import threading
import time
from collections import Counter

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from torchvision import transforms
from ultralytics import YOLO
try:
    from fer import FER
except Exception:  # pragma: no cover
    FER = None

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from src.models.caption_model import CaptionNet  # noqa: E402
from src.utils.text import Vocabulary  # noqa: E402
from src.utils.vision import enhance_frame, filter_small_boxes, open_camera  # noqa: E402

app = Flask(__name__)

state_lock = threading.Lock()
latest_caption = "Initializing..."
latest_detected = "none"
latest_movement = "none"
latest_scene_description = "Analyzing surroundings..."
latest_person_characteristics = "No person detected."
latest_surroundings_characteristics = "Surroundings analysis unavailable."
latest_face_environment = "Face and environment analysis unavailable."
latest_face_emotion = "Emotion analysis unavailable."
latest_nlp_report = "NLP report unavailable."

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


def describe_person_characteristics(person_details):
    if not person_details:
        return "No person detected."
    count = len(person_details)
    closest = max(person_details, key=lambda x: x["area_ratio"])
    moving_people = sum(1 for p in person_details if p["movement"] != "stationary")
    return (
        f"{count} person detected. Closest person is {closest['distance']} at {closest['position']}. "
        f"{moving_people} person appears to be moving."
    )


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


def _get_bundle():
    global _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        caption_model, vocab, max_len = load_caption_model(
            app.config["WEIGHTS"], app.config["VOCAB"], device
        )
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


def _analyze_frame(frame_bgr, client_id="default"):
    global latest_caption, latest_detected, latest_movement, latest_scene_description
    global latest_person_characteristics, latest_surroundings_characteristics
    global latest_face_environment
    global latest_face_emotion
    global latest_nlp_report

    bundle = _get_bundle()
    now = time.time()
    _cleanup_client_state(now)

    frame_for_model = enhance_frame(frame_bgr)

    caption_interval = max(0.1, app.config["CAPTION_INTERVAL"])
    with _client_state_lock:
        last_caption_time = _last_caption_time_by_client.get(client_id, 0.0)
        cached_caption = _last_caption_by_client.get(client_id, "Initializing...")
        previous_centers = _previous_centers_by_client.get(client_id, {})
        _last_seen_by_client[client_id] = now

    if now - last_caption_time >= caption_interval:
        caption = generate_caption(bundle.caption_model, bundle.vocab, frame_for_model, bundle.max_len, bundle.device)
        with _client_state_lock:
            _last_caption_time_by_client[client_id] = now
            _last_caption_by_client[client_id] = caption
        cached_caption = caption

    det_conf = app.config["DET_CONF"]
    det_iou = app.config["DET_IOU"]
    img_size = app.config["IMG_SIZE"]
    min_area_ratio = app.config["MIN_AREA_RATIO"]

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
        valid_indices = filter_small_boxes(results[0], min_area_ratio=min_area_ratio)
        for idx in valid_indices:
            cls_id = cls_list[idx]
            label = names[int(cls_id)]
            labels.append(label)
            x1, y1, x2, y2 = xyxy_list[idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            track_key = f"{label}_{idx}"
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
    nlp_report_text = build_nlp_report(
        cached_caption, person_text, surroundings_text, face_emotion_text, wearables_text
    )

    scene_description_text = build_scene_description(cached_caption, movement_details)

    # Keep existing `/caption` endpoint behavior working by updating globals.
    with state_lock:
        latest_caption = cached_caption
        latest_detected = detected_text
        latest_movement = movement_text
        latest_scene_description = scene_description_text
        latest_person_characteristics = person_text
        latest_surroundings_characteristics = surroundings_text
        latest_face_environment = face_environment_text
        latest_face_emotion = face_emotion_text
        latest_nlp_report = nlp_report_text

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
        (255, 0, 255),
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
        f"Clothes/Accessories: {wearables_text[:70]}",
        (10, 175),
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
        "nlp_report": nlp_report_text,
    }
    return annotated, payload


def generate_frames():
    cap = open_camera(app.config["CAMERA_ID"])
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated, _payload = _analyze_frame(frame, client_id="server_cam")

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/browser_cam")
def browser_cam():
    return render_template("browser_cam.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
        return jsonify({"error": "No frame provided. Send multipart field 'frame' or JSON {image: dataURL}."}), 400

    annotated, payload = _analyze_frame(frame, client_id=client_id)
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return jsonify({"error": "Failed to encode annotated frame."}), 500

    payload["annotated_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify(payload)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    # Alias for `/analyze_frame` (frontend/backends often expect `/predict`)
    if request.method == "OPTIONS":
        return "", 204
    return analyze_frame()


@app.route("/caption")
def caption():
    with state_lock:
        payload = {
            "caption": latest_caption,
            "detected": latest_detected,
            "movement": latest_movement,
            "scene_description": latest_scene_description,
            "person_characteristics": latest_person_characteristics,
            "surroundings_characteristics": latest_surroundings_characteristics,
            "face_environment": latest_face_environment,
            "face_emotion": latest_face_emotion,
            "nlp_report": latest_nlp_report,
        }
    return jsonify(payload)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="artifacts/caption_model.pt")
    parser.add_argument("--vocab", default="artifacts/vocab.json")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--min-area-ratio", type=float, default=0.0015)
    parser.add_argument("--caption-interval", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.config["WEIGHTS"] = args.weights
    app.config["VOCAB"] = args.vocab
    app.config["DETECTOR"] = args.detector
    app.config["CAMERA_ID"] = args.camera_id
    app.config["DET_CONF"] = args.det_conf
    app.config["DET_IOU"] = args.det_iou
    app.config["IMG_SIZE"] = args.img_size
    app.config["MIN_AREA_RATIO"] = args.min_area_ratio
    app.config["CAPTION_INTERVAL"] = args.caption_interval

    if not os.path.exists(args.weights) or not os.path.exists(args.vocab):
        raise FileNotFoundError(
            "Model artifacts not found. Train first to create artifacts/caption_model.pt and artifacts/vocab.json"
        )

    app.run(host=args.host, port=args.port, debug=False)
