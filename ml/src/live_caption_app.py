import argparse
import time

import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

from models.caption_model import CaptionNet
from utils.text import Vocabulary
from utils.vision import enhance_frame, filter_small_boxes, open_camera


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
    tensor = transform(rgb).unsqueeze(0).to(device)
    return tensor


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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_model, vocab, max_len = load_caption_model(args.weights, args.vocab, device)
    detector = YOLO(args.detector)

    cap = open_camera(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    last_caption = ""
    last_caption_time = 0.0
    caption_interval = max(0.1, args.caption_interval)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_for_model = enhance_frame(frame)
        now = time.time()
        if now - last_caption_time >= caption_interval:
            last_caption = generate_caption(caption_model, vocab, frame_for_model, max_len, device)
            last_caption_time = now

        results = detector.predict(
            frame_for_model, conf=args.det_conf, iou=args.det_iou, imgsz=args.img_size, verbose=False
        )
        annotated = results[0].plot()

        labels = []
        if results and len(results[0].boxes) > 0:
            names = results[0].names
            valid_idx = filter_small_boxes(results[0], min_area_ratio=args.min_area_ratio)
            for i in valid_idx:
                cls_id = results[0].boxes.cls.tolist()[i]
                labels.append(names[int(cls_id)])

        label_text = ", ".join(sorted(set(labels))) if labels else "none"
        cv2.putText(
            annotated,
            f"Detected: {label_text}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Caption: {last_caption}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live Detection + Captioning", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to caption model .pt")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--detector", default="yolov8n.pt", help="YOLO model name/path")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=960)
    parser.add_argument("--min-area-ratio", type=float, default=0.0015)
    parser.add_argument("--caption-interval", type=float, default=1.0)
    main(parser.parse_args())
