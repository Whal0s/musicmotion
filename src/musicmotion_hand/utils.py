from __future__ import annotations

from typing import Iterable, Tuple


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def bbox_from_points(points: Iterable[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs = []
    ys = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def center_from_points(points: Iterable[Tuple[int, int]]) -> Tuple[int, int]:
    xs = []
    ys = []
    for x, y in points:
        xs.append(x)
        ys.append(y)
    if not xs:
        return (0, 0)
    return (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))


