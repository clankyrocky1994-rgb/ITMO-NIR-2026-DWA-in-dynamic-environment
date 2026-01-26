# planner_astar.py
# Простой A* планировщик по 2D-сетке для склада

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import heapq
import numpy as np

GridPt = Tuple[int, int]  # (i, j) индекс клетки: i по Y, j по X


@dataclass
class RectObstacle:
    """Прямоугольное препятствие в МИРОВЫХ координатах (x,y)."""
    cx: float
    cy: float
    hx: float  # half-size по X (половина ширины)
    hy: float  # half-size по Y (половина длины)


class AStarGridPlanner:
    """
    Планировщик пути A*.
   
    """

    def __init__(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        resolution: float = 0.25,
        inflate: float = 0.25,
    ):
        # Границы области планирования (склад) в метрах
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)

        # Размер клетки (например 0.25м = 25см)
        self.res = float(resolution)

        # Насколько "раздувать" препятствия (чтобы робот не тёрся об них)
        self.inflate = float(inflate)

        # Сюда складываем препятствия
        self.obstacles: List[RectObstacle] = []

        # Сетка (0 свободно, 1 занято)
        self.grid: Optional[np.ndarray] = None

    # ---------- преобразования мир <-> сетка ----------

    def world_to_grid(self, xy: np.ndarray) -> GridPt:
        """
        Берём точку в мире (x,y) и переводим в индекс клетки (i,j).
        i - строка (ось Y), j - колонка (ось X)
        """
        x, y = float(xy[0]), float(xy[1])
        j = int(round((x - self.x_min) / self.res))
        i = int(round((y - self.y_min) / self.res))
        return (i, j)

    def grid_to_world(self, p: GridPt) -> np.ndarray:
        """
        Берём клетку (i,j) и возвращаем центр клетки в мире (x,y).
        """
        i, j = p
        x = self.x_min + j * self.res
        y = self.y_min + i * self.res
        return np.array([x, y], dtype=float)

    def _in_bounds(self, p: GridPt) -> bool:
        assert self.grid is not None
        H, W = self.grid.shape
        return 0 <= p[0] < H and 0 <= p[1] < W

    # ---------- построение карты препятствий ----------

    def add_rect_obstacle(self, cx: float, cy: float, hx: float, hy: float) -> None:
        """Добавить прямоугольник-объект как препятствие."""
        self.obstacles.append(RectObstacle(cx, cy, hx, hy))

    def build_grid(self) -> np.ndarray:
        """
        Создаём grid: 0 свободно, 1 занято.
        Внутри помечаем клетки, которые попадают в препятствия.
        """
        H = int(round((self.y_max - self.y_min) / self.res)) + 1
        W = int(round((self.x_max - self.x_min) / self.res)) + 1
        grid = np.zeros((H, W), dtype=np.uint8)

        # Для каждого препятствия: пометим клетки, которые внутри его прямоугольника
        for obs in self.obstacles:
            xmin = obs.cx - obs.hx - self.inflate
            xmax = obs.cx + obs.hx + self.inflate
            ymin = obs.cy - obs.hy - self.inflate
            ymax = obs.cy + obs.hy + self.inflate

            # Пробегаем по сетке и помечаем занятые клетки
            for i in range(H):
                y = self.y_min + i * self.res
                if y < ymin or y > ymax:
                    continue
                for j in range(W):
                    x = self.x_min + j * self.res
                    if xmin <= x <= xmax:
                        grid[i, j] = 1

        self.grid = grid
        return grid

    # ---------- A* поиск пути ----------

    def _heuristic(self, a: GridPt, b: GridPt) -> float:
        # Манхэттенское расстояние (дёшево и стабильно)
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors4(self, p: GridPt) -> List[GridPt]:
        i, j = p
        return [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]

    def plan_grid(self, start: GridPt, goal: GridPt) -> Optional[List[GridPt]]:
        """
        A* в координатах СЕТКИ.
        Возвращает список клеток от start до goal, или None если пути нет.
        """
        if self.grid is None:
            raise RuntimeError("Call build_grid() before planning")

        # Если старт/цель на препятствии — сразу не получится
        if not self._in_bounds(start) or not self._in_bounds(goal):
            return None
        if self.grid[start] == 1 or self.grid[goal] == 1:
            return None

        open_heap: List[Tuple[float, float, GridPt]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0.0, start))

        came_from: Dict[GridPt, GridPt] = {}
        gscore: Dict[GridPt, float] = {start: 0.0}
        closed = set()

        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur in closed:
                continue
            if cur == goal:
                # восстановим путь назад
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path

            closed.add(cur)

            for nxt in self._neighbors4(cur):
                if not self._in_bounds(nxt):
                    continue
                if self.grid[nxt] == 1:
                    continue

                ng = g + 1.0  # цена шага (пока просто 1)
                if ng < gscore.get(nxt, 1e9):
                    gscore[nxt] = ng
                    came_from[nxt] = cur
                    f = ng + self._heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f, ng, nxt))

        return None

    def plan(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Планирование в МИРОВЫХ координатах.
        Возвращает список точек (x,y) в мире.
        """
        if self.grid is None:
            self.build_grid()

        start_g = self.world_to_grid(start_xy)
        goal_g = self.world_to_grid(goal_xy)

        path_g = self.plan_grid(start_g, goal_g)
        if path_g is None:
            return None

        # переводим клетки в мировые точки
        path_xy = [self.grid_to_world(p) for p in path_g]
        return path_xy

    def simplify_path(self, path_xy: List[np.ndarray], step: int = 2) -> List[np.ndarray]:
        """
        Прореживание пути: берём каждую step-ю точку.
        Чтобы mocap не получал слишком много микроточек.
        """
        if len(path_xy) <= 2:
            return path_xy
        out = path_xy[::step]
        if (out[-1] != path_xy[-1]).any():
            out.append(path_xy[-1])
        return out

if __name__ == "__main__":
    planner = AStarGridPlanner(
        x_min=-3.0, x_max=3.0,
        y_min=-7.0, y_max=7.0,
        resolution=0.5,
        inflate=0.3,
    )

    # добавим один "стеллаж" в центре
    planner.add_rect_obstacle(cx=0.0, cy=0.0, hx=0.5, hy=1.5)

    grid = planner.build_grid()

    start = np.array([-2.5, -6.0])
    goal  = np.array([ 2.5,  6.0])

    path = planner.plan(start, goal)

    if path is None:
        print("❌ Путь не найден")
    else:
        print(f"✅ Путь найден, точек: {len(path)}")
        for p in path[:5]:
            print(" ", p)
        print(" ...")
        for p in path[-5:]:
            print(" ", p)