import os
import sys
import time

import cv2
import streamlit as st
import torch
from torchvision import transforms
from ultralytics import YOLO

# Allow imports from src/
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from models.caption_model import CaptionNet  # noqa: E402
from utils.text import Vocabulary  # noqa: E402
from utils.vision import enhance_frame, filter_small_boxes, open_camera  # noqa: E402


@st.cache_resource
def load_models(weights_path, vocab_path, detector_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(weights_path, map_location=device)
    caption_model = CaptionNet(
        embed_size=ckpt["embed_size"],
        hidden_size=ckpt["hidden_size"],
        vocab_size=ckpt["vocab_size"],
    ).to(device)
    caption_model.load_state_dict(ckpt["model_state_dict"])
    caption_model.eval()
    detector = YOLO(detector_path)
    return caption_model, vocab, detector, ckpt.get("max_len", 25), device


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
def generate_caption(caption_model, vocab, frame, max_len, device):
    image = frame_to_tensor(frame, device)
    features = caption_model.encoder(image)
    ids = caption_model.decoder.sample(
        features=features,
        max_len=max_len,
        start_id=vocab.stoi["<start>"],
        end_id=vocab.stoi["<end>"],
    )
    text = vocab.decode(ids)
    return text if text else "unable to generate caption"


def main():
    st.set_page_config(page_title="Live Camera Captioning", layout="wide")
    st.title("Live Camera Feed Captioning (CNN + LSTM + YOLO)")

    st.sidebar.header("Model Settings")
    weights = st.sidebar.text_input("Caption model weights", "artifacts/caption_model.pt")
    vocab_path = st.sidebar.text_input("Vocab JSON", "artifacts/vocab.json")
    detector_path = st.sidebar.text_input("YOLO detector", "yolov8n.pt")
    camera_id = st.sidebar.number_input("Camera ID", min_value=0, max_value=5, value=0)
    det_conf = st.sidebar.slider("Detection confidence", 0.1, 0.9, 0.35, 0.05)
    det_iou = st.sidebar.slider("Detection IOU", 0.2, 0.9, 0.45, 0.05)
    img_size = st.sidebar.select_slider("YOLO image size", options=[640, 768, 960, 1280], value=960)
    min_area_ratio = st.sidebar.slider("Min object area ratio", 0.0005, 0.01, 0.0015, 0.0005)
    caption_interval = st.sidebar.slider("Caption update interval (sec)", 0.2, 5.0, 1.0, 0.1)
    frame_delay = st.sidebar.slider("Frame delay (sec)", 0.01, 0.20, 0.03, 0.01)

    if not (os.path.exists(weights) and os.path.exists(vocab_path)):
        st.warning("Train model first so artifacts exist: artifacts/caption_model.pt and artifacts/vocab.json")
        st.stop()

    caption_model, vocab, detector, max_len, device = load_models(weights, vocab_path, detector_path)
    start = st.button("Start Live Camera")
    stop = st.button("Stop")

    if "run_live" not in st.session_state:
        st.session_state.run_live = False
    if start:
        st.session_state.run_live = True
    if stop:
        st.session_state.run_live = False

    image_placeholder = st.empty()
    caption_placeholder = st.empty()

    if st.session_state.run_live:
        cap = open_camera(int(camera_id))
        if not cap.isOpened():
            st.error("Could not open camera.")
            st.session_state.run_live = False
            st.stop()

        last_caption = ""
        last_caption_time = 0.0
        try:
            while st.session_state.run_live:
                ok, frame = cap.read()
                if not ok:
                    st.error("Failed to read frame from camera.")
                    break

                frame_for_model = enhance_frame(frame)
                now = time.time()
                if now - last_caption_time >= caption_interval:
                    last_caption = generate_caption(caption_model, vocab, frame_for_model, max_len, device)
                    last_caption_time = now

                results = detector.predict(
                    frame_for_model,
                    conf=float(det_conf),
                    iou=float(det_iou),
                    imgsz=int(img_size),
                    verbose=False,
                )
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                labels = []
                if results and len(results[0].boxes) > 0:
                    names = results[0].names
                    valid_idx = filter_small_boxes(results[0], min_area_ratio=float(min_area_ratio))
                    for i in valid_idx:
                        cls_id = results[0].boxes.cls.tolist()[i]
                        labels.append(names[int(cls_id)])
                detected = ", ".join(sorted(set(labels))) if labels else "none"

                image_placeholder.image(annotated, channels="RGB", use_container_width=True)
                caption_placeholder.markdown(f"**Detected:** {detected}  \n**Caption:** {last_caption}")
                time.sleep(frame_delay)
        finally:
            cap.release()


if __name__ == "__main__":
    main()
