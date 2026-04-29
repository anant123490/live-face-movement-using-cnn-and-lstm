import cv2


def open_camera(camera_id=0, width=1280, height=720):
    candidates = [
        (camera_id, cv2.CAP_DSHOW),
        (camera_id, cv2.CAP_MSMF),
        (camera_id, cv2.CAP_ANY),
    ]
    for idx, backend in candidates:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return cv2.VideoCapture(camera_id)


def enhance_frame(frame_bgr):
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)
    merged = cv2.merge((y_eq, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.2, blur, -0.2, 0)


def filter_small_boxes(result, min_area_ratio=0.0015):
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return []
    h, w = result.orig_shape
    frame_area = float(h * w)
    filtered = []
    for i, xyxy in enumerate(result.boxes.xyxy.tolist()):
        x1, y1, x2, y2 = xyxy
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if area / frame_area >= min_area_ratio:
            filtered.append(i)
    return filtered
