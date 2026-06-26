from dataclasses import dataclass
from typing import Iterable, Optional, TypeAlias

import numpy as np


# def wrap_angle(angle: float) -> float:
#     """Wrap angle to [-pi, pi]."""
#     return (angle + np.pi) % (2.0 * np.pi) - np.pi
CircleObstacle: TypeAlias = (
    tuple[float, float, float]
    | tuple[float, float, float, float, float]
)

RectObstacle: TypeAlias = tuple[float, float, float, float]

@dataclass
class HolonomicDWAConfig:
    # Simulation
    dt: float = 0.1
    predict_time: float = 1.5
    eta_progress: float = 8.0       # reward for moving closer to goal
    zero_speed_penalty: float = 5.0 # penalty for standing still far from goal
    min_speed_dist: float = 0.5     # apply zero-speed penalty if goal is farther
    obstacle_clearance_cap: float = 0.10

    # Velocity limits for holonomic base
    max_vx: float = 0.55
    max_vy: float = 0.55
    max_w: float = 0.0  # first keep 0 for stability with mocap/IK

    # Acceleration limits
    max_ax: float = 0.8
    max_ay: float = 0.8
    max_aw: float = 1.5

    # Sampling
    vx_samples: int = 7
    vy_samples: int = 7
    w_samples: int = 1

    # Robot size
    robot_radius: float = 0.40
    safety_margin: float = 0.08


    # DWA weights: these will be optimized later
    alpha_goal: float = 2.0      # goal attraction
    beta_obstacle: float = 1.5   # obstacle avoidance
    gamma_speed: float = 0.4     # speed encouragement
    delta_smooth: float = 0.2    # smoothness of commands


class HolonomicDWAPlanner:
    """
    Holonomic Dynamic Window Approach.

    Candidate command:
        u = [vx, vy, omega]

    For the first working version:
        omega can be disabled by max_w=0 and w_samples=1.
    """

    def __init__(self, config: Optional[HolonomicDWAConfig] = None):
        self.cfg = config if config is not None else HolonomicDWAConfig()
        self.last_cmd = np.zeros(3, dtype=float)

    def reset(self):
        self.last_cmd[:] = 0.0

    def plan(
        self,
        state: np.ndarray,
        goal_xy: np.ndarray,
        circle_obstacles: Iterable[CircleObstacle],
        rect_obstacles: Optional[Iterable[RectObstacle]] = None,
    ):
        """
        Parameters
        ----------
        state:
            [x, y] or [x, y, yaw]
        goal_xy:
            [x_goal, y_goal]
        circle_obstacles:
            list of (x, y, radius)
        rect_obstacles:
            list of axis-aligned rectangles:
            (cx, cy, hx, hy), where hx/hy are half sizes

        Returns
        -------
        best_cmd:
            np.array([vx, vy, omega])
        best_traj:
            trajectory predicted for best command, shape [N, 3]
        info:
            debug info with costs
        """

        state = np.asarray(state, dtype=float)
        goal_xy = np.asarray(goal_xy, dtype=float)

        x = float(state[0])
        y = float(state[1])
        yaw = float(state[2]) if len(state) >= 3 else 0.0

        rect_obstacles = list(rect_obstacles) if rect_obstacles is not None else []
        circle_obstacles = list(circle_obstacles)

        vx_range, vy_range, w_range = self._dynamic_window()

        best_cost = float("inf")
        best_cmd = np.zeros(3, dtype=float)
        best_traj = None
        best_info = {}

        # fallback на случай, если ВСЕ траектории считаются collision
        least_bad_clearance = -float("inf")
        least_bad_cmd = np.zeros(3, dtype=float)
        least_bad_traj = None
        least_bad_info = {}

        for vx in vx_range:
            for vy in vy_range:
                for w in w_range:
                    cmd = np.array([vx, vy, w], dtype=float)
                    traj = self._predict_trajectory(x, y, yaw, cmd)

                    cost_info = self._trajectory_cost(
                        traj=traj,
                        cmd=cmd,
                        goal_xy=goal_xy,
                        circle_obstacles=circle_obstacles,
                        rect_obstacles=rect_obstacles,
                    )

                    total_cost = cost_info["total"]

                    # Если траектория collision, обычным победителем она быть не может,
                    # но запоминаем "наименее плохую" по максимальному clearance.
                    if cost_info.get("collision", False):
                        clearance = float(cost_info.get("min_clearance", -float("inf")))

                        if clearance > least_bad_clearance:
                            least_bad_clearance = clearance
                            least_bad_cmd = cmd
                            least_bad_traj = traj
                            least_bad_info = cost_info

                        continue

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_cmd = cmd
                        best_traj = traj
                        best_info = cost_info

        # Если вообще не нашлось безопасной траектории,
        # возвращаем наименее плохую, чтобы робот хотя бы пытался выбраться,
        # а не вставал навсегда с cmd=[0,0,0].
        if best_traj is None and least_bad_traj is not None:
            best_cmd = least_bad_cmd
            best_traj = least_bad_traj
            best_info = least_bad_info


        self.last_cmd = best_cmd.copy()

        return best_cmd, best_traj, best_info

    def _dynamic_window(self):
        c = self.cfg
        vx0, vy0, w0 = self.last_cmd

        vx_min = max(-c.max_vx, vx0 - c.max_ax * c.dt)
        vx_max = min(c.max_vx, vx0 + c.max_ax * c.dt)

        vy_min = max(-c.max_vy, vy0 - c.max_ay * c.dt)
        vy_max = min(c.max_vy, vy0 + c.max_ay * c.dt)

        w_min = max(-c.max_w, w0 - c.max_aw * c.dt)
        w_max = min(c.max_w, w0 + c.max_aw * c.dt)

        vx_range = np.linspace(vx_min, vx_max, c.vx_samples)
        vy_range = np.linspace(vy_min, vy_max, c.vy_samples)

        if c.w_samples <= 1 or c.max_w == 0.0:
            w_range = np.array([0.0])
        else:
            w_range = np.linspace(w_min, w_max, c.w_samples)

        return vx_range, vy_range, w_range

    def _predict_trajectory(
        self,
        x: float,
        y: float,
        yaw: float,
        cmd: np.ndarray,
        ) -> np.ndarray:
        c = self.cfg
        vx, vy, w = cmd

        n_steps = max(1, int(c.predict_time / c.dt))
        traj = np.zeros((n_steps, 3), dtype=float)

        px = x
        py = y
        th = yaw

        for i in range(n_steps):
            px += float(vx) * c.dt
            py += float(vy) * c.dt

            # omega сейчас выключена, но yaw всё равно храним для совместимости
            

            traj[i] = [px, py, th]
        return traj

    def _trajectory_cost(
        self,
        traj: np.ndarray,
        cmd: np.ndarray,
        goal_xy: np.ndarray,
        circle_obstacles: list[CircleObstacle],
        rect_obstacles: list[RectObstacle],
    ):
        c = self.cfg

        final_xy = traj[-1, :2]

        # current position is approximately the first predicted point minus one step
        current_xy = traj[0, :2] - cmd[:2] * self.cfg.dt

        start_goal_dist = float(np.linalg.norm(current_xy - goal_xy))
        goal_dist = float(np.linalg.norm(final_xy - goal_xy))

        # Positive progress means the trajectory moves closer to the goal
        progress = start_goal_dist - goal_dist

        min_clearance = self._min_clearance(
            traj=traj,
            circle_obstacles=circle_obstacles,
            rect_obstacles=rect_obstacles,
        )

        # Collision or too close
        if min_clearance <= 0.0:
            return {
                "total": float("inf"),
                "goal": goal_dist,
                "obstacle": float("inf"),
                "speed": 0.0,
                "smooth": 0.0,
                "min_clearance": min_clearance,
                "collision": True,
            }

        obstacle_cost = 1.0 / (min_clearance + 0.02)
        near_collision_cost = 0.0
        if min_clearance < 0.10:
            near_collision_cost = 20.0 * (0.10 - min_clearance)

        speed = float(np.linalg.norm(cmd[:2]))
        max_speed = float(np.linalg.norm([c.max_vx, c.max_vy]))
        speed_cost = max_speed - speed

        smooth_cost = np.linalg.norm(cmd - self.last_cmd)

        zero_cost = 0.0
        if start_goal_dist > c.min_speed_dist and speed < 0.03:
            zero_cost = c.zero_speed_penalty

        total = float(
            c.alpha_goal * goal_dist
            + c.beta_obstacle * obstacle_cost
            + c.gamma_speed * speed_cost
            + c.delta_smooth * smooth_cost
            - c.eta_progress * progress
            + zero_cost
            + near_collision_cost
        )

        return {
            "progress": progress,
            "zero": zero_cost,
            "total": total,
            "goal": goal_dist,
            "obstacle": obstacle_cost,
            "speed": speed_cost,
            "smooth": smooth_cost,
            "min_clearance": min_clearance,
            "collision": False,
            "near_collision": near_collision_cost,
        }

    def _min_clearance(
        self,
        traj: np.ndarray,
        circle_obstacles: list[CircleObstacle],
        rect_obstacles: list[RectObstacle],
    ) -> float:
        c = self.cfg
        min_clearance = float("inf")

        for i, p in enumerate(traj[:, :2]):
            future_t = (i + 1) * c.dt

            # Circular obstacles.
            # Поддерживаем два формата:
            # (x, y, radius)
            # (x, y, radius, vx, vy)
            for obs in circle_obstacles:
                if len(obs) == 5:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])
                    ovx = float(obs[3])
                    ovy = float(obs[4])

                    ox = ox + ovx * future_t
                    oy = oy + ovy * future_t

                elif len(obs) == 3:
                    ox = float(obs[0])
                    oy = float(obs[1])
                    radius = float(obs[2])

                else:
                    raise ValueError(
                        f"Circle obstacle must be (x, y, r) or (x, y, r, vx, vy), got: {obs}"
                    )

                center_dist = float(
                    np.linalg.norm(p - np.array([ox, oy], dtype=float))
                )

                clearance = center_dist - radius - c.robot_radius - c.safety_margin
                min_clearance = min(min_clearance, clearance)

            # Rectangular obstacles / walls.
            for rect in rect_obstacles:
                clearance = self._clearance_to_rect(p, rect)
                clearance -= c.robot_radius + c.safety_margin
                min_clearance = min(min_clearance, clearance)

        return min_clearance




    @staticmethod
    def _clearance_to_rect(
        point_xy: np.ndarray,
        rect: RectObstacle,
    ) -> float:
        """
        Signed distance from point to axis-aligned rectangle.

        rect = (cx, cy, hx, hy)
        hx, hy are half sizes.

        Positive: outside rectangle.
        Negative: inside rectangle.
        """
        cx, cy, hx, hy = rect

        dx = abs(point_xy[0] - cx) - hx
        dy = abs(point_xy[1] - cy) - hy

        outside_dx = max(dx, 0.0)
        outside_dy = max(dy, 0.0)

        outside_dist = float(np.linalg.norm([outside_dx, outside_dy]))

        if dx <= 0.0 and dy <= 0.0:
            # Point is inside rectangle
            inside_dist = min(hx - abs(point_xy[0] - cx), hy - abs(point_xy[1] - cy))
            return -inside_dist

        return outside_dist