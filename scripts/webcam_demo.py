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

from musicmotion_hand.detector import HandPositionDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Webcam hand position detector demo.")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    ap.add_argument("--width", type=int, default=1280, help="Capture width (best effort)")
    ap.add_argument("--height", type=int, default=720, help="Capture height (best effort)")
    ap.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect")
    ap.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal mirroring (default is mirrored/selfie mode)",
    )
    args = ap.parse_args()

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

    with HandPositionDetector(max_num_hands=args.max_hands) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            hands = detector.detect(frame)
            frame = detector.draw(frame, hands, draw_landmarks=True)

            # Small HUD
            cv2.putText(
                frame,
                f"hands: {len(hands)} | press q to quit",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("musicmotion - hand detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


