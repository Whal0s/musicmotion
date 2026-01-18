from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np

# Allow running without installing the package (repo-local usage).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from musicmotion_hand.detector import HandPositionDetector  # noqa: E402
from musicmotion_hand.types import HandLandmark, HandPosition  # noqa: E402


ASCII_LOGO = [
                                                              
    r"                         _                      _   _         ",
    r" _ __ ___  _   _ ___(_) ___ _ __ ___   ___ | |_(_) ___  _ __  ",
    r"| '_ ` _ \| | | / __| |/ __| '_ ` _ \ / _ \| __| |/ _ \| '_ \ ",
    r"| | | | | | |_| \__ \ | (__| | | | | | (_) | |_| | (_) | | | |",
    r"|_| |_| |_|\__,_|___/_|\___|_| |_| |_|\___/ \__|_|\___/|_| |_|",
]


def draw_ascii_banner(canvas_bgr: np.ndarray, title_lines=ASCII_LOGO) -> None:
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = (18, 18, 18)  # dark background

    # Retro-ish green phosphor + subtle scanline effect.
    green = (40, 255, 120)
    shadow = (0, 40, 0)

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1.15
    thickness = 2

    # Use fixed-width "cell" placement to preserve ASCII alignment (OpenCV fonts aren't truly monospace).
    char_w = cv2.getTextSize("M", font, scale, thickness)[0][0]
    line_h = cv2.getTextSize("A", font, scale, thickness)[0][1] + 8
    block_h = line_h * len(title_lines)
    y0 = max(8, (h - block_h) // 2)

    # Left padding so it reads like a header strip.
    x0 = 14

    for i, line in enumerate(title_lines):
        y = y0 + (i + 1) * line_h
        # Draw character-by-character to keep the ASCII art columns aligned.
        for j, ch in enumerate(line):
            x = x0 + j * char_w
            # Shadow (heavier + offset) for a retro drop-shadow look
            cv2.putText(canvas_bgr, ch, (x + 3, y + 3), font, scale, shadow, thickness + 4, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x, y), font, scale, green, thickness, cv2.LINE_AA)

    # Scanlines
    for y in range(0, h, 4):
        canvas_bgr[y : y + 1, :] = (12, 12, 12)

    # Bottom border line
    cv2.line(canvas_bgr, (0, h - 1), (w - 1, h - 1), (60, 60, 60), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Retro UI camera window + hand joint overlay.")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width (best effort)")
    ap.add_argument("--height", type=int, default=720, help="Capture height (best effort)")
    ap.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect")
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring (default is mirrored/selfie mode)",
    )
    ap.add_argument("--banner-height", type=int, default=92, help="Top strip height in pixels")
    ap.add_argument(
        "--tasks-model",
        default="models/hand_landmarker.task",
        help="Path to MediaPipe Tasks model (auto-downloaded if missing)",
    )
    args = ap.parse_args()

    # On macOS, AVFoundation is usually the most reliable backend and is also what
    # triggers the system camera permission prompt for the launching app.
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        msg = (
            f"Could not open camera index {args.camera}.\n\n"
            "If you're on macOS and you saw 'not authorized to capture video', grant Camera access to the app\n"
            "you launched this from (Terminal / iTerm / Cursor) in:\n"
            "  System Settings -> Privacy & Security -> Camera\n\n"
            "Then quit and re-run this script.\n"
        )
        raise RuntimeError(msg)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "musicmotion"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    banner = np.zeros((args.banner_height, args.width, 3), dtype=np.uint8)
    draw_ascii_banner(banner)

    last_t = time.time()
    fps = 0.0

    with HandPositionDetector(max_num_hands=args.max_hands, tasks_model_path=args.tasks_model) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Detect on the raw (unmirrored) frame so handedness labels remain correct.
            hands = detector.detect(frame)

            # Prepare display frame. If mirroring, mirror coordinates too (so overlay lines up).
            if not args.no_mirror:
                display = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                hands_to_draw = _mirror_hands(hands, width_px=fw)
            else:
                display = frame
                hands_to_draw = hands

            display = detector.draw(display, hands_to_draw, draw_landmarks=True)

            # FPS (simple exponential smoothing)
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst = 1.0 / dt
            fps = 0.85 * fps + 0.15 * inst if fps > 0 else inst
            last_t = now

            # Resize banner to current frame width (camera may not match requested width).
            fh, fw = display.shape[:2]
            if banner.shape[1] != fw:
                banner = np.zeros((args.banner_height, fw, 3), dtype=np.uint8)
                draw_ascii_banner(banner)

            hud = banner.copy()
            cv2.putText(
                hud,
                f"hands: {len(hands)} | fps: {fps:0.1f} | q/esc quit",
                (14, hud.shape[0] - 12),
                cv2.FONT_HERSHEY_PLAIN,
                1.4,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out = np.zeros((hud.shape[0] + fh, fw, 3), dtype=np.uint8)
            out[: hud.shape[0], :, :] = hud
            out[hud.shape[0] :, :, :] = display

            cv2.imshow(window_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


def _mirror_hands(hands: list[HandPosition], *, width_px: int) -> list[HandPosition]:
    """
    Mirror HandPosition coordinates horizontally for drawing on a horizontally flipped frame.

    This keeps handedness labels untouched while making points/lines line up with the mirrored image.
    """

    out: list[HandPosition] = []
    for hand in hands:
        lms: list[HandLandmark] = []
        for lm in hand.landmarks:
            x_px = int(width_px - 1 - lm.x_px)
            lms.append(
                HandLandmark(
                    idx=lm.idx,
                    x_norm=1.0 - lm.x_norm,
                    y_norm=lm.y_norm,
                    z_norm=lm.z_norm,
                    x_px=x_px,
                    y_px=lm.y_px,
                )
            )

        x0, y0, x1, y1 = hand.bbox_px
        bbox = (int(width_px - 1 - x1), y0, int(width_px - 1 - x0), y1)
        cx, cy = hand.center_px
        center = (int(width_px - 1 - cx), cy)
        tips = {k: (int(width_px - 1 - v[0]), v[1]) for k, v in hand.fingertips_px.items()}

        out.append(
            HandPosition(
                handedness_label=hand.handedness_label,
                handedness_score=hand.handedness_score,
                landmarks=lms,
                bbox_px=bbox,
                center_px=center,
                fingertips_px=tips,
            )
        )

    return out


if __name__ == "__main__":
    raise SystemExit(main())

