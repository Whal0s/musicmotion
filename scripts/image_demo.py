from __future__ import annotations

import argparse
import os
import sys

import cv2

# Allow running without installing the package (repo-local usage).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from musicmotion_hand.detector import HandPositionDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Image hand position detector demo.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--out", required=True, help="Path to output image (annotated)")
    ap.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to detect")
    args = ap.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    with HandPositionDetector(static_image_mode=True, max_num_hands=args.max_hands) as detector:
        hands = detector.detect(frame)
        out = detector.draw(frame, hands, draw_landmarks=True)

    ok = cv2.imwrite(args.out, out)
    if not ok:
        raise RuntimeError(f"Could not write output image: {args.out}")

    print(f"hands: {len(hands)}")
    for i, h in enumerate(hands):
        print(
            f"[{i}] {h.handedness_label} score={h.handedness_score} "
            f"center={h.center_px} bbox={h.bbox_px} index_tip={h.fingertips_px.get('index')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


