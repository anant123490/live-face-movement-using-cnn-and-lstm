#!/usr/bin/env python3
"""
Run the Keras-only emotion HTTP API (same idea as a small Flask + load_model script).

  python run_emotion_api.py
  python run_emotion_api.py --model emotion_model.hdf5 --host 0.0.0.0 --port 5000

Endpoints: GET /  -> plain text "Emotion Detection API Running"
            POST /predict -> JSON {"image": <nested array>} or {"image_b64": "..."}

Implementation lives in emotion_server.py (lazy model load, CORS, health, Gunicorn-ready).
"""

from __future__ import annotations

import emotion_server as es


if __name__ == "__main__":
    args = es.parse_args()
    es._apply_config_from_args(args)
    es.app.run(host=args.host, port=args.port, debug=False)
