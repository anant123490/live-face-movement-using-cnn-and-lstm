"""
Minimal Flask API: Keras HDF5 emotion model only (no YOLO, no captioning).

Example:
  python emotion_server.py --model emotion_model.hdf5 --host 0.0.0.0 --port 5001

POST /predict (JSON):
  {"image_b64": "<jpeg/png base64>"}   # recommended
  {"image": [[...], ...]}               # nested uint8 array (BGR or grayscale)

Optional:
  {"use_whole_frame": true|false}       # true = same preprocessing as the common classroom snippet
                                         # (BGR->gray, resize full frame to 48x48). false = Haar face crop when possible.
                                         # If you omit this and send only {"image": [...]}, whole-frame mode is used
                                         # so old clients match without changes. For {"image_b64": ...} the default is false.

Requires: tensorflow, opencv-python-headless, flask, numpy
"""

from __future__ import annotations

import argparse
import base64
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)


def _truthy(val):
    if val is None:
        return False
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "on")


@app.errorhandler(RequestEntityTooLarge)
def _too_large(_e):
    return (
        jsonify({"error": "Request body too large. Send a smaller image or raise MAX_UPLOAD_MB."}),
        413,
    )


@app.after_request
def _cors(response):
    if app.config.get("ENABLE_CORS"):
        response.headers["Access-Control-Allow-Origin"] = app.config.get("CORS_ORIGIN", "*")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "86400"
    return response

_model = None
_face_cascade = None


def _labels():
    raw = app.config.get("EMOTION_LABELS", "")
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


def _load_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model as keras_load_model

        path = app.config["MODEL_PATH"]
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        _model = keras_load_model(path)
    return _model


def _haar():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def _decode_frame(data):
    if isinstance(data.get("image"), list):
        img = np.array(data["image"], dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    if isinstance(data.get("image_b64"), str):
        raw_b64 = data["image_b64"]
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        raw = base64.b64decode(raw_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    return None


def _frame_to_face_input(frame_bgr, use_whole_frame):
    if frame_bgr.ndim == 2:
        gray_full = frame_bgr
        frame_bgr = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    if use_whole_frame:
        gray = gray_full
    else:
        faces = _haar().detectMultiScale(gray_full, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            h, w = gray_full.shape[:2]
            side = min(h, w)
            y0 = max(0, (h - side) // 2)
            x0 = max(0, (w - side) // 2)
            gray = gray_full[y0 : y0 + side, x0 : x0 + side]
        else:
            fx, fy, fw, fh = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            gray = gray_full[int(fy) : int(fy + fh), int(fx) : int(fx + fw)]

    face = cv2.resize(gray, (48, 48))
    face = face.astype(np.float32) / 255.0
    return np.reshape(face, (1, 48, 48, 1))


@app.route("/")
def home():
    return "Emotion Detection API Running"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "emotion-only"})


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    try:
        model = _load_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500

    data = request.get_json(silent=True) or {}
    frame = _decode_frame(data)
    if frame is None:
        return jsonify({"error": "Provide JSON with 'image_b64' or 'image' array."}), 400

    if "use_whole_frame" in data:
        use_whole = bool(data.get("use_whole_frame"))
    elif isinstance(data.get("image"), list) and not isinstance(data.get("image_b64"), str):
        # Match typical class snippets: request.json['image'] as nested uint8 array, full frame -> 48x48.
        use_whole = True
    else:
        use_whole = False
    try:
        face_in = _frame_to_face_input(frame, use_whole_frame=use_whole)
        preds = model.predict(face_in, verbose=0)
        logits = np.asarray(preds[0]).flatten()
        idx = int(np.argmax(logits))
        labels = _labels()
        idx = min(idx, len(labels) - 1) if labels else 0
        emotion = labels[idx] if labels else str(idx)
        conf = float(np.max(logits))
        return jsonify({"emotion": emotion, "confidence": conf})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.getenv("EMOTION_MODEL", "emotion_model.hdf5"))
    p.add_argument(
        "--labels",
        default=os.getenv(
            "EMOTION_LABELS",
            "Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral",
        ),
    )
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "5001")))
    p.add_argument(
        "--enable-cors",
        action="store_true",
        default=_truthy(os.getenv("ENABLE_CORS", "0")),
    )
    p.add_argument("--cors-origin", default=os.getenv("CORS_ORIGIN", "*"))
    p.add_argument(
        "--max-upload-mb",
        type=int,
        default=int(os.getenv("MAX_UPLOAD_MB", "15")),
    )
    args, _unknown = p.parse_known_args()
    return args


def _apply_config_from_args(args):
    app.config["MODEL_PATH"] = args.model
    app.config["EMOTION_LABELS"] = args.labels
    app.config["ENABLE_CORS"] = bool(args.enable_cors)
    app.config["CORS_ORIGIN"] = (args.cors_origin or "*").strip() or "*"
    max_mb = max(1, int(args.max_upload_mb))
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024


_apply_config_from_args(parse_args())


if __name__ == "__main__":
    args = parse_args()
    _apply_config_from_args(args)
    app.run(host=args.host, port=args.port, debug=False)
