## musicmotion — Hand position detector (Python)

This repo includes a small **hand position detector** built on **MediaPipe Hands** that outputs:

- 21 hand landmarks (normalized + pixel coordinates)
- handedness (`Left` / `Right`) + confidence
- bounding box + center point
- fingertip pixel positions (thumb/index/middle/ring/pinky)

### Setup

Create a venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Optional (installs the package in editable mode):

```bash
pip install -e .
```

### Webcam demo

```bash
python scripts/webcam_demo.py
```

### Retro UI webcam demo (ASCII banner)

```bash
python3 scripts/retro_ui_demo.py
```

Controls:
- `q`: quit

### Title screen (hand-controlled BPM + key mode)

```bash
python3 scripts/title_screen.py
```

### Troubleshooting

- **`source: no such file or directory: .venv/bin/activate`**: you haven’t created the venv yet. Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- **`ModuleNotFoundError: No module named 'cv2'`**: OpenCV isn’t installed in your active environment. Run:

```bash
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -c "import cv2; print('cv2 ok', cv2.__version__)"
```

- **Python 3.13 note**: if `pip install` fails for `opencv-python` or `mediapipe`, use **Python 3.12** for this project (wheel availability can vary by OS/Python version).
- **macOS camera permission**: if you see `OpenCV: not authorized to capture video`, enable Camera access for the app you launched from:
  - `System Settings -> Privacy & Security -> Camera` (enable Terminal/iTerm/Cursor)
  - If you don’t get prompted, you can reset the permission and re-run:

```bash
tccutil reset Camera
```

- **MediaPipe missing `mp.solutions`**: some installs don’t include the classic Hands API. This project will fall back to the **MediaPipe Tasks HandLandmarker** API, which requires a model file:
  - The script will **auto-download** `models/hand_landmarker.task` on first run (or see `models/README.md` for manual download)
- **SSL certificate download error** (`CERTIFICATE_VERIFY_FAILED`): if the model auto-download fails on macOS, it’s usually your Python SSL cert store. The code will fall back to `curl`, but you can also:
  - Run the “Install Certificates.command” that comes with python.org Python
  - Or manually download the model via `curl` (see `models/README.md`)

### Image demo

```bash
python scripts/image_demo.py --image /path/to/image.jpg --out /path/to/output.jpg
```

### Use as a library

```python
import cv2
from musicmotion_hand.detector import HandPositionDetector

detector = HandPositionDetector(max_num_hands=2)

frame = cv2.imread("image.jpg")
hands = detector.detect(frame)
frame = detector.draw(frame, hands)
cv2.imwrite("out.jpg", frame)
```


