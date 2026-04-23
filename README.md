# Sign Language Detection Using YOLOv5

Real-time sign language detection from webcam or video using a custom-trained YOLOv5 model, with optional audio feedback and ESP32 camera support.

## Features

- Real-time sign language detection via webcam, video, or ESP32 camera stream
- Audio feedback using text-to-speech (pyttsx3) — announces detected signs aloud
- On-screen overlays: FPS, confidence threshold, audio status, detection count
- Keyboard controls during live detection
- CSV export of predictions

## Detected Signs

`Hello` · `Yes` · `No` · `Thanks` · `I Love You` · `Please`

## Project Structure

```
├── detect.py              # Core YOLOv5 detection script
├── run.py                 # Quick launcher (webcam, conf=0.1)
├── runmod.py              # Detection + audio feedback + ESP32 support
├── runmod2.py             # Detection + audio feedback (webcam only)
├── capture_image.py       # Collect training images from webcam
├── best.pt                # Trained YOLOv5 model weights
├── yolov5/                # YOLOv5 submodule
└── Data/Sign_language_data/  # Dataset (train/test + data.yaml)
```

## Requirements

```bash
pip install torch torchvision opencv-python pyttsx3 ultralytics
pip install -r yolov5/requirements.txt
```

## Usage

### Quick start (webcam)
```bash
python run.py
```

### Detection with audio feedback
```bash
python runmod2.py --weights best.pt --source 0 --conf-thres 0.85
```

### Detection with audio + ESP32 camera
```bash
python runmod.py --weights best.pt --esp32-url http://<ESP32_IP>:81/stream
```

### Detection only (no audio)
```bash
python detect.py --weights best.pt --source 0 --conf-thres 0.25
```

### Collect training images
```bash
python capture_image.py
```

## Keyboard Controls (live detection)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `m` | Mute / unmute audio |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

## Training

The model was trained on a custom sign language dataset located in `Data/Sign_language_data/`. See `Sign_language_Generation_Using_YOLO_v5.ipynb` for the full training notebook.

## Model

- Architecture: YOLOv5s
- Weights: `best.pt` (custom trained)
- Input size: 416×416 (quick run) / 640×640 (default)
- Confidence threshold: 0.85 (audio scripts) / 0.1 (quick run)
