#!/usr/bin/env python3
"""
Webcam demo with audio playback based on right hand position.

The vertical position of the right hand controls the pitch of the tone.
Higher hand position = higher pitch.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys

import cv2

# Allow running without installing the package (repo-local usage).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from musicmotion_hand.audio import SCALES, ToneGenerator  # noqa: E402
from musicmotion_hand.detector import HandPositionDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Webcam hand position detector with audio playback."
    )
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument(
        "--width", type=int, default=1280, help="Capture width (best effort)"
    )
    ap.add_argument(
        "--height", type=int, default=720, help="Capture height (best effort)"
    )
    ap.add_argument(
        "--max-hands", type=int, default=2, help="Maximum number of hands to detect"
    )
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring (default is mirrored/selfie mode)",
    )
    ap.add_argument(
        "--scale",
        type=str,
        default="pentatonic",
        choices=list(SCALES.keys()),
        help="Musical scale to use (default: pentatonic)",
    )
    ap.add_argument(
        "--volume",
        type=float,
        default=0.3,
        help="Volume level (0.0 to 1.0, default: 0.3)",
    )
    args = ap.parse_args()
    
    # Default position when no hand is detected (middle of scale)
    DEFAULT_POSITION = 0.5

    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "On macOS: System Settings -> Privacy & Security -> Camera -> allow your terminal/Cursor."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    with HandPositionDetector(max_num_hands=args.max_hands) as detector, ToneGenerator(
        scale=args.scale, volume=args.volume
    ) as tone_gen:
        print(f"Using scale: {args.scale}")
        print("Move your RIGHT hand up and down to change pitch")
        print("Press 'q' or ESC to quit")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            hands = detector.detect(frame)
            frame = detector.draw(frame, hands, draw_landmarks=True)

            # Find right hand and update audio
            right_hand = None
            for hand in hands:
                if hand.handedness_label == "Right":
                    right_hand = hand
                    break

            if right_hand is not None:
                # Use normalized Y position from center of hand
                frame_height = frame.shape[0]
                # Guard against division by zero with invalid frames
                if frame_height > 0:
                    y_norm = right_hand.center_px[1] / frame_height
                    tone_gen.set_position(y_norm, active=True)

                    # Visual feedback for active note
                    cv2.putText(
                        frame,
                        f"Playing: Y={y_norm:.2f}",
                        (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    tone_gen.set_position(DEFAULT_POSITION, active=False)
            else:
                # No right hand detected - silence
                tone_gen.set_position(DEFAULT_POSITION, active=False)
                cv2.putText(
                    frame,
                    "No right hand detected",
                    (12, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            # HUD
            cv2.putText(
                frame,
                f"hands: {len(hands)} | scale: {args.scale} | press q to quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("musicmotion - audio demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
