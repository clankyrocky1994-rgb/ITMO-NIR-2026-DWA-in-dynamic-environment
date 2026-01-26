from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


def wrap_pi(a: float) -> float:
    """Нормализует угол в диапазон [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_from_last_segment(path_xy: List[np.ndarray]) -> Optional[float]:
    """
    'Умный yaw' без ручных GOAL_TH:
    Берём последние две точки пути и считаем направление последнего отрезка.
    Возвращает угол (рад) или None, если пути мало.
    """
    if path_xy is None or len(path_xy) < 2:
        return None
    a = path_xy[-2]
    b = path_xy[-1]
    yaw = float(np.arctan2(b[1] - a[1], b[0] - a[0]))
    return wrap_pi(yaw)


@dataclass
class YawAligner:
    """
    Хранит желаемый yaw и решает, когда его применять.
    """
    desired_yaw: Optional[float] = None
    # начинать докручивать за N точек до конца пути
    start_on_last_k: int = 2

    def update_from_path(self, path_xy: List[np.ndarray]) -> None:
        self.desired_yaw = yaw_from_last_segment(path_xy)

    def should_apply(self, path_idx: int, path_len: int) -> bool:
        if self.desired_yaw is None:
            return False
        if path_len <= 0:
            return False
        # например: начинаем на последних 2 точках
        return path_idx >= max(0, path_len - self.start_on_last_k)