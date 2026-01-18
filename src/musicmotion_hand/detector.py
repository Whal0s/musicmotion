from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2

from .types import HandLandmark, HandPosition
from .utils import bbox_from_points, center_from_points, clamp_int
from .model_assets import ensure_hand_landmarker_task


@dataclass
class _SmoothingState:
    center_px: Optional[Tuple[int, int]] = None


HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # index
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # middle
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # ring
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    # palm base
    (0, 17),
]


@dataclass(frozen=True)
class _SolutionsBackend:
    mp: object
    hands: object


@dataclass(frozen=True)
class _TasksBackend:
    mp: object
    landmarker: object
    running_mode_video: object


def _try_create_solutions_backend(
    static_image_mode: bool,
    max_num_hands: int,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> Optional[_SolutionsBackend]:
    import mediapipe as mp  # type: ignore

    if not hasattr(mp, "solutions"):
        return None
    hands_mod = mp.solutions.hands
    hands = hands_mod.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return _SolutionsBackend(mp=mp, hands=hands)


def _try_create_tasks_backend(
    model_path: str,
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> _TasksBackend:
    """
    Fallback for MediaPipe distributions that do not include `mp.solutions`.

    Uses the MediaPipe Tasks HandLandmarker API, which requires a `.task` model asset on disk.
    """

    import mediapipe as mp  # type: ignore

    # Import locations can differ slightly across builds.
    try:
        from mediapipe.tasks.python import BaseOptions  # type: ignore
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode  # type: ignore
    except Exception:  # pragma: no cover
        from mediapipe.tasks import python as mp_python  # type: ignore

        BaseOptions = mp_python.BaseOptions
        vision = mp_python.vision
        HandLandmarker = vision.HandLandmarker
        HandLandmarkerOptions = vision.HandLandmarkerOptions
        RunningMode = vision.RunningMode

    model_path = ensure_hand_landmarker_task(model_path)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = HandLandmarker.create_from_options(options)
    return _TasksBackend(mp=mp, landmarker=landmarker, running_mode_video=RunningMode.VIDEO)


class HandPositionDetector:
    """
    Hand position detector using MediaPipe Hands.

    Input frames are expected as **BGR** images (OpenCV default).
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_centers_alpha: Optional[float] = 0.4,
        tasks_model_path: str = "models/hand_landmarker.task",
    ) -> None:
        self._solutions: Optional[_SolutionsBackend] = _try_create_solutions_backend(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._tasks: Optional[_TasksBackend] = None
        self._tasks_timestamp_ms = 0
        self._static_image_mode = static_image_mode
        self._max_num_hands = max_num_hands

        if self._solutions is None:
            # Fall back to Tasks API.
            try:
                self._tasks = _try_create_tasks_backend(
                    model_path=tasks_model_path,
                    max_num_hands=max_num_hands,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    "MediaPipe does not provide `mp.solutions` in your environment, so this project uses the\n"
                    "MediaPipe Tasks HandLandmarker fallback, which needs a model file on disk:\n"
                    f"  {tasks_model_path}\n\n"
                    "Download the model and try again (see README Troubleshooting / Models section)."
                ) from e
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "Could not initialize MediaPipe Hands.\n"
                    "Your installed `mediapipe` package does not expose `mp.solutions`, and the Tasks fallback\n"
                    "could not be initialized.\n\n"
                    "Run this and paste the output:\n"
                    "  python3 -c \"import mediapipe as mp; print(mp.__file__); print(getattr(mp,'__version__',None)); print([a for a in dir(mp) if not a.startswith('_')][:80])\""
                ) from e

        self._smooth_centers_alpha = smooth_centers_alpha
        self._smooth_state = _SmoothingState()

    def close(self) -> None:
        if self._solutions is not None:
            self._solutions.hands.close()
        if self._tasks is not None:
            try:
                self._tasks.landmarker.close()
            except Exception:
                pass

    def __enter__(self) -> "HandPositionDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def detect(self, frame_bgr) -> List[HandPosition]:
        h, w = frame_bgr.shape[:2]

        if self._solutions is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._solutions.hands.process(frame_rgb)

            if not results.multi_hand_landmarks:
                self._smooth_state.center_px = None
                return []

            handedness_list = results.multi_handedness or []
            positions: List[HandPosition] = []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label: Optional[str] = None
                score: Optional[float] = None
                if i < len(handedness_list) and handedness_list[i].classification:
                    c = handedness_list[i].classification[0]
                    label = getattr(c, "label", None)
                    score = float(getattr(c, "score", 0.0))

                positions.append(self._build_hand_position(hand_landmarks.landmark, label, score, w, h))

            return positions

        if self._tasks is None:
            return []

        # Tasks API path
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp = self._tasks.mp
        if not hasattr(mp, "Image") or not hasattr(mp, "ImageFormat"):
            raise RuntimeError("Your MediaPipe build does not expose `mp.Image` required for the Tasks API.")

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Tasks VIDEO mode requires monotonically increasing timestamps.
        self._tasks_timestamp_ms += 33  # ~30fps; good enough for stable tracking
        ts = self._tasks_timestamp_ms

        if self._static_image_mode:
            # For static images, detect() is okay if available; otherwise video with ts works.
            try:
                result = self._tasks.landmarker.detect(mp_image)
            except Exception:
                result = self._tasks.landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._tasks.landmarker.detect_for_video(mp_image, ts)

        hand_landmarks_list = getattr(result, "hand_landmarks", None) or []
        handedness_list = getattr(result, "handedness", None) or []

        if not hand_landmarks_list:
            self._smooth_state.center_px = None
            return []

        positions: List[HandPosition] = []
        for i, landmarks in enumerate(hand_landmarks_list):
            label = None
            score = None
            if i < len(handedness_list) and handedness_list[i]:
                cat0 = handedness_list[i][0]
                label = getattr(cat0, "category_name", None) or getattr(cat0, "display_name", None)
                score = float(getattr(cat0, "score", 0.0))

            positions.append(self._build_hand_position(landmarks, label, score, w, h))

        return positions

    def _build_hand_position(self, landmarks, label: Optional[str], score: Optional[float], w: int, h: int) -> HandPosition:
        lm_px: List[HandLandmark] = []
        pts_px: List[Tuple[int, int]] = []
        for idx, lm in enumerate(landmarks):
            x_px = clamp_int(int(round(float(lm.x) * w)), 0, w - 1)
            y_px = clamp_int(int(round(float(lm.y) * h)), 0, h - 1)
            lm_px.append(
                HandLandmark(
                    idx=idx,
                    x_norm=float(lm.x),
                    y_norm=float(lm.y),
                    z_norm=float(getattr(lm, "z", 0.0)),
                    x_px=x_px,
                    y_px=y_px,
                )
            )
            pts_px.append((x_px, y_px))

        bbox_px = bbox_from_points(pts_px)
        center_px = center_from_points(pts_px)
        center_px = self._maybe_smooth_center(center_px)

        tips = {
            "thumb": (lm_px[4].x_px, lm_px[4].y_px),
            "index": (lm_px[8].x_px, lm_px[8].y_px),
            "middle": (lm_px[12].x_px, lm_px[12].y_px),
            "ring": (lm_px[16].x_px, lm_px[16].y_px),
            "pinky": (lm_px[20].x_px, lm_px[20].y_px),
        }

        return HandPosition(
            handedness_label=label,
            handedness_score=score,
            landmarks=lm_px,
            bbox_px=bbox_px,
            center_px=center_px,
            fingertips_px=tips,
        )

    def _maybe_smooth_center(self, center_px: Tuple[int, int]) -> Tuple[int, int]:
        a = self._smooth_centers_alpha
        if a is None:
            return center_px
        if not (0.0 <= a <= 1.0):
            return center_px
        prev = self._smooth_state.center_px
        if prev is None:
            self._smooth_state.center_px = center_px
            return center_px
        cx = int(round((1 - a) * prev[0] + a * center_px[0]))
        cy = int(round((1 - a) * prev[1] + a * center_px[1]))
        smoothed = (cx, cy)
        self._smooth_state.center_px = smoothed
        return smoothed

    def draw(self, frame_bgr, hands: List[HandPosition], draw_landmarks: bool = True):
        if draw_landmarks:
            for hand in hands:
                # Lines
                for a, b in HAND_CONNECTIONS:
                    if a < len(hand.landmarks) and b < len(hand.landmarks):
                        p0 = (hand.landmarks[a].x_px, hand.landmarks[a].y_px)
                        p1 = (hand.landmarks[b].x_px, hand.landmarks[b].y_px)
                        cv2.line(frame_bgr, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
                # Points
                for lm in hand.landmarks:
                    cv2.circle(frame_bgr, (lm.x_px, lm.y_px), 3, (40, 255, 120), -1, lineType=cv2.LINE_AA)

        for hand in hands:
            x0, y0, x1, y1 = hand.bbox_px
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.circle(frame_bgr, hand.center_px, 6, (0, 0, 255), -1)

            label = hand.handedness_label or "Hand"
            if hand.handedness_score is not None:
                label = f"{label} {hand.handedness_score:.2f}"
            cv2.putText(
                frame_bgr,
                label,
                (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for name, pt in hand.fingertips_px.items():
                cv2.circle(frame_bgr, pt, 6, (255, 0, 0), -1)
                cv2.putText(
                    frame_bgr,
                    name,
                    (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame_bgr


