# Live Camera Image Captioning (CNN + LSTM + Detection)

This project captures frames from a live camera feed, detects objects, and generates scene captions.

## Features

- Real-time camera capture with OpenCV
- Object detection using YOLOv8
- Caption generation using CNN encoder + LSTM decoder
- On-frame overlay for detected objects and generated captions

## Project Structure

- `requirements.txt` - Python dependencies
- `requirements-sagemaker.txt` - SageMaker-specific dependencies
- `serve_sagemaker.sh` - Gunicorn startup script for SageMaker container
- `flask_app.py` - Flask website for live camera captioning
- `templates/index.html` - Flask web page for video and caption display
- `streamlit_app.py` - Streamlit web UI for live camera captioning
- `src/create_sample_dataset.py` - Create a sample image-caption dataset
- `src/train_caption.py` - Train CNN-LSTM caption model
- `src/live_caption_app.py` - Live webcam captioning + detection
- `src/models/caption_model.py` - Encoder/decoder model definitions
- `src/utils/text.py` - Vocabulary, tokenization, decoding utilities

## Setup

1. Create virtual environment:

```bash
python -m venv .venv
```

2. Activate environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare Data

Use any caption dataset (Flickr8k, Flickr30k, COCO). Create a CSV file with:

- `image_path` : image file path
- `caption` : text caption

Example:

```csv
image_path,caption
data/images/1.jpg,a man riding a bicycle
data/images/2.jpg,two dogs playing in grass
```

Or generate a sample dataset in this project:

```bash
python src/create_sample_dataset.py --samples 120
```

This creates:

- `data/sample_images/*.png`
- `data/captions.csv`

## Train Caption Model

```bash
python src/train_caption.py --csv data/captions.csv --epochs 10 --batch-size 32
```

Model artifacts are saved in `artifacts/`.

## Run Live Camera App

```bash
python src/live_caption_app.py --weights artifacts/caption_model.pt --vocab artifacts/vocab.json
```

Press `q` to quit.

For clearer detections in noisy scenes:

```bash
python src/live_caption_app.py --weights artifacts/caption_model.pt --vocab artifacts/vocab.json --det-conf 0.45 --det-iou 0.45 --img-size 960 --min-area-ratio 0.0015
```

## Run Streamlit Website

```bash
streamlit run streamlit_app.py
```

Then open the shown local URL (usually `http://localhost:8501`) and click **Start Live Camera**.

## Run Flask Website

```bash
python flask_app.py --weights artifacts/caption_model.pt --vocab artifacts/vocab.json
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## EC2 Deployment (Browser Webcam + Backend API)

EC2 instances do not have a physical webcam, so the correct setup is:
- **Frontend** (in your browser): captures webcam frames via `getUserMedia()`
- **Backend** (on EC2): receives frames over HTTP and runs ML on them (`/predict`)

### Install dependencies on EC2

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ec2.txt
```

Note: `requirements.txt` uses `opencv-python-headless` which is suitable for servers.

### Run the EC2 app

Use the new entrypoint `app.py`:

```bash
python app.py --weights artifacts/caption_model.pt --vocab artifacts/vocab.json --host 0.0.0.0 --port 5000
```

Or with Gunicorn (loads `wsgi.py`, which applies weights/vocab from env or CLI — safe with Gunicorn’s own flags via `parse_known_args`):

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

- **API discovery**: `GET /api` — JSON list of routes and a sample `curl` for `/predict`.
- **CORS** (e.g. static site calling your EC2 IP): `ENABLE_CORS=1` or `--enable-cors`; optional `--cors-origin https://your-site.com`.
- **Upload limit**: `MAX_UPLOAD_MB` or `--max-upload-mb` (default 25) sets Flask `MAX_CONTENT_LENGTH`.

### Access from your laptop (recommended: SSH tunnel)

Browsers only allow webcam access on **HTTPS** or **localhost**. Easiest is to tunnel EC2 → localhost:

```bash
ssh -i your-key.pem -L 5000:127.0.0.1:5000 ubuntu@<EC2_PUBLIC_IP>
```

Then open:
- `http://localhost:5000/` (or `http://localhost:5000/browser_cam`)

### Alternative: HTTPS on EC2

If you want to access it directly by public IP/domain, terminate TLS (HTTPS) using a reverse proxy (e.g. Nginx + Let’s Encrypt).

### Optional: Keras HDF5 emotion model (`emotion_model.hdf5`)

If you have a classic Keras FER-style model (input shape `(1, 48, 48, 1)`), place `emotion_model.hdf5` on the server and run:

```bash
export KERAS_EMOTION_MODEL=/path/to/emotion_model.hdf5
python app.py --keras-emotion-model "$KERAS_EMOTION_MODEL"
```

- Full pipeline `/predict` returns extra fields: `keras_emotion`, `keras_emotion_label`, `keras_emotion_confidence`.
- Standalone JSON endpoint: `POST /predict_keras_emotion` with `{"image_b64": "..."}` (recommended) or `{"image": <nested array>}`.

### Emotion-only microservice (`emotion_server.py`)

Tiny Flask app with **no YOLO and no caption model** — only `emotion_model.hdf5` + `POST /predict`.

Dependencies (minimal): `flask`, `numpy`, `opencv-python-headless`, `tensorflow`.

```bash
pip install flask numpy opencv-python-headless tensorflow
python emotion_server.py --model emotion_model.hdf5 --host 0.0.0.0 --port 5001
# Or: gunicorn -b 0.0.0.0:5001 emotion_wsgi:app
```

- `GET /` → plain text: `Emotion Detection API Running`
- `GET /health` → `{"status":"ok","service":"emotion-only"}`
- `POST /predict` → JSON `{"image_b64":"..."}` or `{"image":[...]}`. Optional `"use_whole_frame": true` to mimic a naive full-frame resize (default uses Haar face crop when possible).

### Production setup (systemd + Nginx)

Files included:
- `deploy/ec2/ml-camera-ml.service` (systemd service)
- `deploy/ec2/nginx-ml-camera.conf` (Nginx site)

Example commands on Ubuntu:

```bash
sudo apt update
sudo apt install -y nginx

# Nginx site
sudo cp deploy/ec2/nginx-ml-camera.conf /etc/nginx/sites-available/ml-camera
sudo ln -sf /etc/nginx/sites-available/ml-camera /etc/nginx/sites-enabled/ml-camera
sudo nginx -t
sudo systemctl restart nginx

# systemd service (edit paths if your project lives elsewhere)
sudo cp deploy/ec2/ml-camera-ml.service /etc/systemd/system/ml-camera-ml.service
sudo systemctl daemon-reload
sudo systemctl enable --now ml-camera-ml
sudo systemctl status ml-camera-ml --no-pager
```

## Notes

- This starter is designed for clarity and extensibility.
- For higher accuracy, train longer and tune vocabulary size, embedding size, and hidden dimensions.
- For higher FPS, use a lighter encoder backbone or run captioning every N frames.

## SageMaker Requirements

For AWS SageMaker/container deployments, use:

```bash
pip install -r requirements-sagemaker.txt
```

Then start the app on endpoint port `8080`:

```bash
bash serve_sagemaker.sh
```

Important:
- SageMaker endpoints do not provide direct local webcam access.
- Use uploaded images/video frames or client-side camera capture sent to your API.
