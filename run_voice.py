# ===== CRITICAL FIXES (TOP) =====
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import platform
import pathlib

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "yolov5")
sys.path.append(BASE_DIR)
sys.path.append(YOLO_PATH)
# =================================

import argparse
import torch
import pyttsx3
import threading
import time

from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import cv2, non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device, smart_inference_mode

# ========== AUDIO ==========
last_spoken = {}
COOLDOWN_SECONDS = 4.0
is_speaking = False  # global flag to prevent overlap

def speak_now(text):
    """Runs in its own thread. Reinits engine each call — most reliable approach."""
    global is_speaking
    try:
        is_speaking = True
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)
        engine.say(f"I see {text}")
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"TTS Error: {e}")
    finally:
        is_speaking = False

def speak_detection(label, confidence):
    global is_speaking

    current_time = time.time()

    # Skip if still speaking
    if is_speaking:
        print(f"⏭ Skipping '{label}' — still speaking")
        return

    # Skip if this label was spoken recently
    if label in last_spoken and current_time - last_spoken[label] < COOLDOWN_SECONDS:
        remaining = COOLDOWN_SECONDS - (current_time - last_spoken[label])
        print(f"⏳ Cooldown for '{label}': {remaining:.1f}s left")
        return

    last_spoken[label] = current_time

    if confidence > 0.9:
        text = f"{label} with high confidence"
    elif confidence > 0.5:
        text = label
    else:
        text = f"possible {label}"

    print(f"🔊 Speaking: '{text}' (conf: {confidence:.2f})")

    # Fire and forget in a new thread
    t = threading.Thread(target=speak_now, args=(text,), daemon=True)
    t.start()

# ========== MAIN ==========
@smart_inference_mode()
def run(weights="best.pt", source="0", conf_thres=0.4):

    device = select_device("")
    model = DetectMultiBackend(weights, device=device)

    stride, names = model.stride, model.names
    imgsz = check_img_size((640, 640), s=stride)

    dataset = LoadStreams(source, img_size=imgsz, stride=stride)

    model.warmup(imgsz=(1, 3, *imgsz))

    print("\n🚀 Running...\n")

    for path, im, im0s, vid_cap, s in dataset:

        im = torch.from_numpy(im).to(model.device).float() / 255
        if len(im.shape) == 3:
            im = im[None]

        pred = model(im)
        pred = non_max_suppression(pred, conf_thres, 0.45)

        for i, det in enumerate(pred):
            im0 = im0s[i].copy()
            annotator = Annotator(im0)

            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Sort by confidence, pick the best detection
                det_sorted = det[det[:, 4].argsort(descending=True)]
                top_det = det_sorted[0]
                top_conf = float(top_det[4])
                top_label = names[int(top_det[5])]

                print(f"👁 Top detection: {top_label} ({top_conf:.2f})")

                if top_conf > conf_thres:
                    speak_detection(top_label, top_conf)

                for *xyxy, conf, cls in det:
                    label = f"{names[int(cls)]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(int(cls), True))

            else:
                print("👁 No detections this frame")

            im0 = annotator.result()
            cv2.imshow("Detection", im0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 Exiting...")
                return

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best.pt")
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--conf-thres", type=float, default=0.4)
    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)