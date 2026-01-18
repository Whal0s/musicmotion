from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

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
    r"                  _                            _   _           ",
    r" _ __ ___  _   _ | | ___  _ __ ___   ___  ___ | |_(_) ___  _ __ ",
    r"| '_ ` _ \| | | || |/ __|| '_ ` _ \ / _ \/ __|| __| |/ _ \| '_ \\",
    r"| | | | | | |_| || |\__ \| | | | | | (_) \__ \| |_| | (_) | | | |",
    r"|_| |_| |_|\__,_||_||___/|_| |_| |_|\___/|___/ \__|_|\___/|_| |_|",
]


def draw_ascii_banner(canvas_bgr: np.ndarray, title_lines=ASCII_LOGO) -> None:
    h, w = canvas_bgr.shape[:2]
    canvas_bgr[:] = (18, 18, 18)

    green = (40, 255, 120)
    shadow = (0, 40, 0)

    font = cv2.FONT_HERSHEY_PLAIN
    scale = 1.15
    thickness = 2

    char_w = cv2.getTextSize("M", font, scale, thickness)[0][0]
    line_h = cv2.getTextSize("A", font, scale, thickness)[0][1] + 8
    block_h = line_h * len(title_lines)
    y0 = max(8, (h - block_h) // 2)
    x0 = 14

    for i, line in enumerate(title_lines):
        y = y0 + (i + 1) * line_h
        for j, ch in enumerate(line):
            x = x0 + j * char_w
            cv2.putText(canvas_bgr, ch, (x + 3, y + 3), font, scale, shadow, thickness + 4, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas_bgr, ch, (x, y), font, scale, green, thickness, cv2.LINE_AA)

    for y in range(0, h, 4):
        canvas_bgr[y : y + 1, :] = (12, 12, 12)
    cv2.line(canvas_bgr, (0, h - 1), (w - 1, h - 1), (60, 60, 60), 1)


def _mirror_hands(hands: List[HandPosition], *, width_px: int) -> List[HandPosition]:
    out: List[HandPosition] = []
    for hand in hands:
        lms: List[HandLandmark] = []
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


def _right_hand(hands: List[HandPosition]) -> Optional[HandPosition]:
    # Prefer explicit Right hand; fall back to first hand.
    for h in hands:
        if (h.handedness_label or "").lower() == "right":
            return h
    return hands[0] if hands else None


def _hand_openness_score(hand: HandPosition) -> float:
    """
    0.0 = closed fist-ish, 1.0 = open-ish.
    Uses fingertip-to-MCP distances normalized by bbox diagonal.
    """

    lms = hand.landmarks
    if len(lms) < 21:
        return 1.0
    x0, y0, x1, y1 = hand.bbox_px
    diag = max(1.0, float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5))

    pairs = [
        (8, 5),  # index tip -> index mcp
        (12, 9),  # middle
        (16, 13),  # ring
        (20, 17),  # pinky
        (4, 2),  # thumb tip -> thumb mcp-ish
    ]
    dsum = 0.0
    for a, b in pairs:
        dx = float(lms[a].x_px - lms[b].x_px)
        dy = float(lms[a].y_px - lms[b].y_px)
        dsum += (dx * dx + dy * dy) ** 0.5
    davg = dsum / len(pairs)
    norm = davg / diag

    # Map a rough range into 0..1 for easier thresholds.
    # Typical: closed ~0.10-0.25, open ~0.40-0.70 (varies).
    return float(np.clip((norm - 0.18) / (0.55 - 0.18), 0.0, 1.0))


@dataclass
class DirectionGate:
    """
    Stabilizes direction gestures (to reduce finicky behavior):
    - requires the same direction for N consecutive frames
    - applies a cooldown between triggers
    """

    last_dir: Optional[str] = None
    stable_count: int = 0
    last_trigger_t: float = 0.0

    def update_and_trigger(self, direction: Optional[str], *, stable_frames: int, cooldown_s: float) -> Optional[str]:
        now = time.time()
        if direction is None:
            self.last_dir = None
            self.stable_count = 0
            return None

        if direction == self.last_dir:
            self.stable_count += 1
        else:
            self.last_dir = direction
            self.stable_count = 1

        if self.stable_count >= stable_frames and (now - self.last_trigger_t) >= cooldown_s:
            self.last_trigger_t = now
            self.stable_count = 0
            return direction

        return None


HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    r: float


def _update_and_draw_particles(frame, particles: List[Particle], dt: float) -> None:
    if not particles:
        return
    out: List[Particle] = []
    for p in particles:
        p.life -= dt
        if p.life <= 0:
            continue
        p.x += p.vx * dt
        p.y += p.vy * dt
        p.vx *= 0.98
        p.vy *= 0.98
        out.append(p)

        a = float(np.clip(p.life / 0.45, 0.0, 1.0))
        col = (int(255 * a), int(255 * a), int(255 * a))
        cv2.circle(frame, (int(p.x), int(p.y)), int(max(1.0, p.r)), col, -1, lineType=cv2.LINE_AA)
    particles[:] = out


def _spawn_hand_particles(particles: List[Particle], hand: HandPosition, n: int = 10) -> None:
    if len(hand.landmarks) < 21:
        return
    emit_idxs = [0, 4, 8, 12, 16, 20]
    for _ in range(n):
        idx = emit_idxs[int(np.random.randint(0, len(emit_idxs)))]
        lm = hand.landmarks[idx]
        jitter = np.random.randn(2) * 3.0
        vx, vy = (np.random.randn() * 10.0, np.random.randn() * 10.0 - 12.0)
        particles.append(
            Particle(
                x=float(lm.x_px + jitter[0]),
                y=float(lm.y_px + jitter[1]),
                vx=float(vx),
                vy=float(vy),
                life=float(0.25 + np.random.rand() * 0.35),
                r=float(1.0 + np.random.rand() * 1.8),
            )
        )


def _draw_hand_bbox_alpha(frame, hand: HandPosition, *, alpha: float = 0.16) -> None:
    x0, y0, x1, y1 = hand.bbox_px
    _draw_rect_alpha(frame, (x0, y0, x1, y1), (255, 255, 255), alpha=alpha)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1, cv2.LINE_AA)


def _draw_retro_hand(frame, hand: HandPosition) -> None:
    for a, b in HAND_CONNECTIONS:
        if a >= len(hand.landmarks) or b >= len(hand.landmarks):
            continue
        p0 = (hand.landmarks[a].x_px, hand.landmarks[a].y_px)
        p1 = (hand.landmarks[b].x_px, hand.landmarks[b].y_px)
        cv2.line(frame, p0, p1, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.line(frame, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)

    for lm in hand.landmarks:
        cv2.circle(frame, (lm.x_px, lm.y_px), 5, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (lm.x_px, lm.y_px), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)


def _point_direction(hand: HandPosition) -> Optional[str]:
    """
    Infer a simple pointing direction from index finger vector.
    Uses index MCP (5) -> index TIP (8).
    """
    lms = hand.landmarks
    if len(lms) < 9:
        return None

    x0, y0 = float(lms[5].x_px), float(lms[5].y_px)
    x1, y1 = float(lms[8].x_px), float(lms[8].y_px)
    vx, vy = x1 - x0, y1 - y0
    mag = (vx * vx + vy * vy) ** 0.5
    if mag < 35.0:
        return None
    vx /= mag
    vy /= mag

    # Cheap "pointing" heuristic: index extension should be >= middle extension.
    if len(lms) >= 13:
        mx0, my0 = float(lms[9].x_px), float(lms[9].y_px)
        mx1, my1 = float(lms[12].x_px), float(lms[12].y_px)
        mlen = ((mx1 - mx0) ** 2 + (my1 - my0) ** 2) ** 0.5
        if mag < mlen * 0.95:
            return None

    if vy < -0.78 and abs(vx) < 0.55:
        return "up"
    if vy > 0.78 and abs(vx) < 0.55:
        return "down"
    if vx < -0.78 and abs(vy) < 0.55:
        return "left"
    if vx > 0.78 and abs(vy) < 0.55:
        return "right"
    return None


def _draw_rect_alpha(frame, rect, color_bgr, alpha: float) -> None:
    """Alpha-blend a solid rect on top of the frame."""
    x0, y0, x1, y1 = rect
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(frame.shape[1], int(x1))
    y1 = min(frame.shape[0], int(y1))
    if x1 <= x0 or y1 <= y0:
        return

    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:, :] = color_bgr
    cv2.addWeighted(overlay, float(alpha), roi, float(1.0 - alpha), 0.0, dst=roi)


def _draw_panel(frame, rect, title: str, active: bool, selected_idx: int, items: List[str]) -> None:
    x0, y0, x1, y1 = rect
    bg = (28, 28, 28)
    border = (40, 255, 120) if active else (80, 80, 80)
    _draw_rect_alpha(frame, rect, bg, alpha=0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), border, 2)

    cv2.putText(frame, title, (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Items list
    inner_top = y0 + 48
    row_h = 34
    for i, text in enumerate(items):
        yy = inner_top + i * row_h
        if yy + row_h > y1 - 10:
            break
        is_sel = i == selected_idx
        if is_sel:
            _draw_rect_alpha(frame, (x0 + 10, yy - 24, x1 - 10, yy + 8), (40, 255, 120), alpha=0.88)
            color = (10, 10, 10)
        else:
            color = (220, 220, 220)
        cv2.putText(frame, text, (x0 + 18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def _draw_info_panel(frame, rect, bpm: int, mode: str) -> None:
    x0, y0, x1, y1 = rect
    _draw_rect_alpha(frame, rect, (20, 20, 20), alpha=0.62)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 60, 60), 2)
    cv2.putText(frame, "PROJECT", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    lines = [
        ("Time", "4/4"),
        ("Loop", "8 bars"),
        ("BPM", str(bpm)),
        ("Key", mode),
        ("Next", "Instrument select (TODO)"),
        ("Ctrl", "Right hand"),
    ]
    yy = y0 + 62
    for k, v in lines:
        cv2.putText(frame, f"{k}:", (x0 + 12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 170, 170), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{v}", (x0 + 90, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        yy += 28


def main() -> int:
    ap = argparse.ArgumentParser(description="MusicMotion title screen (hand-controlled).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--banner-height", type=int, default=92)
    ap.add_argument("--no-mirror", action="store_true", help="Disable mirror (default is selfie mirror).")
    ap.add_argument("--tasks-model", default="models/hand_landmarker.task")
    args = ap.parse_args()

    # Camera
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "musicmotion - title"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    banner = np.zeros((args.banner_height, args.width, 3), dtype=np.uint8)
    draw_ascii_banner(banner)

    # State
    bpm_values = list(range(60, 181, 5))
    bpm_idx = bpm_values.index(120) if 120 in bpm_values else 0
    mode_items = ["Major", "Minor"]
    mode_idx = 0

    active_panel = "bpm"  # or "mode"
    dir_gate = DirectionGate()
    last_t = time.time()
    fps = 0.0
    particles: List[Particle] = []

    with HandPositionDetector(max_num_hands=1, tasks_model_path=args.tasks_model) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            hands_raw = detector.detect(frame)

            # Mirrored display + coordinate mirroring (keeps handedness label stable).
            if not args.no_mirror:
                display = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                hands = _mirror_hands(hands_raw, width_px=fw)
            else:
                display = frame
                hands = hands_raw

            # Choose the right hand for control.
            rh = _right_hand(hands)

            # Update fps
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst = 1.0 / dt
            fps = 0.85 * fps + 0.15 * inst if fps > 0 else inst
            last_t = now

            # Layout
            H, W = display.shape[:2]
            info_w = int(W * 0.24)
            pad = 12
            top = pad
            bottom = H - pad
            left_w = int((W - info_w - 3 * pad) * 0.55)
            right_w = (W - info_w - 3 * pad) - left_w

            bpm_rect = (pad, top, pad + left_w, bottom)
            mode_rect = (pad + left_w + pad, top, pad + left_w + pad + right_w, bottom)
            info_rect = (W - info_w - pad, top, W - pad, bottom)

            # Controls:
            # - point LEFT/RIGHT: choose active control (BPM vs KEY MODE)
            # - point UP/DOWN: adjust the active control
            if rh is not None:
                direction = _point_direction(rh)
                trig = dir_gate.update_and_trigger(direction, stable_frames=3, cooldown_s=0.22)

                if trig == "left":
                    active_panel = "bpm"
                elif trig == "right":
                    active_panel = "mode"
                elif trig == "up":
                    if active_panel == "bpm":
                        bpm_idx = int(np.clip(bpm_idx + 1, 0, len(bpm_values) - 1))
                    else:
                        # Explicit up/down mapping for mode to feel deterministic.
                        mode_idx = 0  # Major
                elif trig == "down":
                    if active_panel == "bpm":
                        bpm_idx = int(np.clip(bpm_idx - 1, 0, len(bpm_values) - 1))
                    else:
                        mode_idx = 1  # Minor

                # Retro hand overlay: white lines + particles + translucent bbox
                _draw_hand_bbox_alpha(display, rh, alpha=0.14)
                _draw_retro_hand(display, rh)
                _spawn_hand_particles(particles, rh, n=8)
            else:
                dir_gate.last_dir = None
                dir_gate.stable_count = 0

            _update_and_draw_particles(display, particles, dt=dt)

            # Render panels
            bpm_items = [f"{bpm} BPM" for bpm in bpm_values]
            _draw_panel(display, bpm_rect, "BPM", active_panel == "bpm", bpm_idx, bpm_items)
            _draw_panel(display, mode_rect, "KEY MODE", active_panel == "mode", mode_idx, mode_items)
            _draw_info_panel(display, info_rect, bpm=bpm_values[bpm_idx], mode=mode_items[mode_idx])

            # Banner & HUD
            if banner.shape[1] != W:
                banner = np.zeros((args.banner_height, W, 3), dtype=np.uint8)
                draw_ascii_banner(banner)
            hud = banner.copy()
            cv2.putText(
                hud,
                "point LEFT/RIGHT: choose control | point UP/DOWN: adjust | q/esc quit",
                (14, hud.shape[0] - 12),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"fps: {fps:0.1f}",
                (W - 140, hud.shape[0] - 12),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            out = np.zeros((hud.shape[0] + H, W, 3), dtype=np.uint8)
            out[: hud.shape[0], :, :] = hud
            out[hud.shape[0] :, :, :] = display

            cv2.imshow(window_name, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


