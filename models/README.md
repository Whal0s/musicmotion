## MediaPipe Tasks model (hand_landmarker.task)

Some MediaPipe Python builds (notably newer ones that don’t expose `mp.solutions`) require the **Tasks API**
to run hand landmark detection. In that mode, you need to download the model file:

- **File**: `models/hand_landmarker.task`

### Download

From the repo root:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

Then run:

```bash
python3 scripts/retro_ui_demo.py --mirror
```


