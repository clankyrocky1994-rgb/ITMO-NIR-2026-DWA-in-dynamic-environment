from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from dwa import HolonomicDWAPlanner, HolonomicDWAConfig

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_XML = _PROJECT_ROOT / "assets" / "stanford_tidybot" / "scene.xml"


# ============================================================
# РУЧНЫЕ ПРОМЕЖУТОЧНЫЕ ЦЕЛИ
# ============================================================

WAYPOINTS = [
    # конец первого прямого коридора
    np.array([-2.40, 1.45], dtype=float),

    # после поворота, уже в горизонтальном участке
    np.array([1.35, 1.75], dtype=float),

    # финальная цель
    np.array([1.35, 4.75], dtype=float),
]


# ============================================================
# СТЕНЫ КОРИДОРА ДЛЯ DWA
# Формат: (cx, cy, hx, hy)
# cx, cy — центр прямоугольника
# hx, hy — половина ширины/длины
# Эти значения соответствуют стенам в scene.xml.
# ============================================================

WALL_RECTS = [
    # bottom wall behind robot
    (-2.40, -6.85, 1.35, 0.08),

    # first vertical corridor
    (-3.65, -2.00, 0.08, 4.95),
    (-1.15, -3.05, 0.08, 3.90),

    # right turn
    (-1.70, 2.85, 2.05, 0.08),
    (0.65, 0.75, 1.90, 0.08),

    # final vertical corridor
    (0.25, 4.10, 0.08, 1.35),
    (2.45, 3.05, 0.08, 2.40),

    # end wall behind goal
    (1.35, 5.35, 1.25, 0.08),
]


# ============================================================
# ДИНАМИЧЕСКИЕ ЦИЛИНДРЫ
# В scene.xml у тебя препятствия — это body obs1 ... obs7,
# внутри которых лежат cylinder-geom'ы.
# ============================================================

DYNAMIC_OBSTACLES = [
    {
        "body": "obs1",
        "base": np.array([-2.40, -4.40], dtype=float),
        "axis": "x",
        "amp": 0.45,
        "omega": 0.70,
        "phase": 0.0,
        "radius": 0.12,
    },
    {
        "body": "obs3",
        "base": np.array([-2.70, -2.00], dtype=float),
        "axis": "x",
        "amp": 0.45,
        "omega": 0.90,
        "phase": 1.0,
        "radius": 0.12,
    },
    {
        "body": "obs5",
        "base": np.array([-2.70, 0.60], dtype=float),
        "axis": "x",
        "amp": 0.45,
        "omega": 0.80,
        "phase": 2.0,
        "radius": 0.12,
    },
    {
        "body": "obs6",
        "base": np.array([-0.35, 1.85], dtype=float),
        "axis": "y",
        "amp": 0.40,
        "omega": 0.75,
        "phase": 0.5,
        "radius": 0.11,
    },
    {
        "body": "obs7",
        "base": np.array([1.25, 3.25], dtype=float),
        "axis": "x",
        "amp": 0.40,
        "omega": 0.85,
        "phase": 1.5,
        "radius": 0.11,
    },
]


@dataclass
class KeyCallback:
    run: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.run = not self.run
            print("run =", self.run)

        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause
            print("pause =", self.pause)

def record_obstacle_traces(model: mujoco.MjModel, obstacle_traces: dict):
    """
    Сохраняет текущие позиции динамических препятствий.
    obstacle_traces:
        {
            "obs1": [np.array([x, y]), ...],
            "obs3": [np.array([x, y]), ...],
            ...
        }
    """

    for obs in DYNAMIC_OBSTACLES:
        body_name = obs["body"]
        body_id = model.body(body_name).id

        pos_xy = model.body_pos[body_id, :2].copy()
        obstacle_traces[body_name].append(pos_xy)


def update_dynamic_obstacles(model: mujoco.MjModel, sim_time: float):
    circle_obstacles = []

    for obs in DYNAMIC_OBSTACLES:
        body_id = model.body(obs["body"]).id

        base = obs["base"].copy()
        axis = obs["axis"]
        amp = float(obs["amp"])
        omega = float(obs["omega"])
        phase = float(obs["phase"])
        radius = float(obs["radius"])

        s = np.sin(omega * sim_time + phase)
        c = np.cos(omega * sim_time + phase)

        pos = base.copy()
        vel = np.zeros(2, dtype=float)

        if axis == "x":
            pos[0] = base[0] + amp * s
            vel[0] = amp * omega * c
        elif axis == "y":
            pos[1] = base[1] + amp * s
            vel[1] = amp * omega * c

        model.body_pos[body_id, 0] = pos[0]
        model.body_pos[body_id, 1] = pos[1]

        circle_obstacles.append((pos[0], pos[1], radius, vel[0], vel[1]))
    
    static_cylinders = [
        (-1.75, -3.20, 0.12),
        (-1.75, -0.80, 0.12),
    ]

    circle_obstacles.extend(static_cylinders)

    return circle_obstacles


def plot_navigation_result(
    trajectory_xy,
    waypoints,
    wall_rects,
    dynamic_obstacles=None,
    goal_xy=None,
    obstacle_traces=None,
    save_path="navigation_path.png",
):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Стены
    for cx, cy, hx, hy in wall_rects:
        rect = Rectangle(
            (cx - hx, cy - hy),
            2 * hx,
            2 * hy,
            alpha=0.5,
        )
        ax.add_patch(rect)

    # Путь робота
    if len(trajectory_xy) > 0:
        traj = np.array(trajectory_xy)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=2, label="Robot path")

        # старт
        ax.scatter(traj[0, 0], traj[0, 1], s=80, marker="o", label="Start")

        # финиш
        ax.scatter(traj[-1, 0], traj[-1, 1], s=80, marker="x", label="Finish")

    # Waypoints
    if waypoints is not None and len(waypoints) > 0:
        wp = np.array(waypoints)
        ax.scatter(wp[:, 0], wp[:, 1], s=60, marker="s", label="Waypoints")

    # Финальная цель
    if goal_xy is not None:
        ax.scatter(goal_xy[0], goal_xy[1], s=120, marker="*", label="Goal")

    # Базовые позиции препятствий
    if dynamic_obstacles is not None:
        for obs in dynamic_obstacles:
            base = obs["base"]
            radius = obs["radius"]
            circ = Circle((base[0], base[1]), radius, fill=False, linestyle="--")
            ax.add_patch(circ)

        # Следы динамических препятствий
    if obstacle_traces is not None:
        for body_name, points in obstacle_traces.items():
            if len(points) < 2:
                continue

            pts = np.array(points)

            ax.plot(
                pts[:, 0],
                pts[:, 1],
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                label=f"{body_name} trace",
            )

            # последняя позиция препятствия
            ax.scatter(
                pts[-1, 0],
                pts[-1, 1],
                s=35,
                marker="o",
            )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Robot trajectory")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

def main():
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # ========================================================
    # Joints, которыми управляет mink
    # ========================================================

    joint_names = [
        "joint_x",
        "joint_y",
        "joint_th",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "joint_7",
    ]

    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    fingers_id = model.actuator("fingers_actuator").id
    GRIP_OPEN = 0

    # ========================================================
    # MINK CONFIGURATION
    # ========================================================

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    # ВАЖНО:
    # первые 3 dof — база, их не фиксируем.
    # руке даём стоимость побольше, чтобы она не болталась,
    # а движение в основном выполнялось базой.
    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 0.15

    posture_task = mink.PostureTask(model, cost=posture_cost)

    tasks = [
        end_effector_task,
        posture_task,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # ========================================================
    # DWA CONFIG
    # ========================================================

    dwa_cfg = HolonomicDWAConfig(
        dt=0.08,
        predict_time=1.5,

        max_vx=0.55,
        max_vy=0.55,
        max_w=0.0,

        max_ax=0.8,
        max_ay=0.8,
        max_aw=1.5,

        vx_samples=9,
        vy_samples=9,
        w_samples=1,

        robot_radius=0.40,
        safety_margin=0.10,

        alpha_goal=2.0,
        beta_obstacle=1.6,
        gamma_speed=0.35,
        delta_smooth=0.25,

        eta_progress=8.0,
        zero_speed_penalty=5.0,
    )

    dwa = HolonomicDWAPlanner(dwa_cfg)

    waypoint_idx = 0
    reach_dist = 0.35

    key_callback = KeyCallback()
    rate = RateLimiter(frequency=200.0, warn=False)

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # ====================================================
        # RESET В HOME
        # ====================================================

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Ставим mocap target ровно в pinch_site.
        # Это та же логика, что была у тебя раньше.
        mink.move_mocap_to_frame(
            model,
            data,
            "pinch_site_target",
            "pinch_site",
            "site",
        )

        mujoco.mj_forward(model, data)

        mocap_id = model.body("pinch_site_target").mocapid[0]

        print("=== DWA + mink navigation ===")
        print("ENTER  - start / stop")
        print("SPACE  - pause")
        print("A*     - not used")
        print("FSM    - not used")

        nav_time = 0.0
        control_timer = 0.0
        current_cmd = np.zeros(3, dtype=float)
        desired_mocap_pos = data.mocap_pos[mocap_id].copy()
        emergency_timer = 0.0
        emergency_cmd = np.zeros(3, dtype=float)
        plot_done = False
        trajectory_xy = []
        obstacle_traces = {
            obs["body"]: []
            for obs in DYNAMIC_OBSTACLES
        }

        while viewer.is_running():
            dt = float(rate.dt)

            if not key_callback.pause:
                nav_time += dt

            sim_time = nav_time


            # Открытый схват, чтобы рука просто была в спокойном состоянии.
            data.ctrl[fingers_id] = GRIP_OPEN

            # 1. Двигаем динамические цилиндры.
            circle_obstacles = update_dynamic_obstacles(model, sim_time)
            record_obstacle_traces(model, obstacle_traces)

            # 2. Текущая цель из списка ручных точек.
            if waypoint_idx >= len(WAYPOINTS):
                current_goal = WAYPOINTS[-1]
            else:
                current_goal = WAYPOINTS[waypoint_idx]

            # 3. Текущее состояние базы.
            # joint_x, joint_y, joint_th находятся в начале qpos.
            base_x = float(data.qpos[0])
            base_y = float(data.qpos[1])
            base_yaw = float(data.qpos[2])

            base_xy = np.array([base_x, base_y], dtype=float)

            trajectory_xy.append(base_xy.copy())

            # 4. Если дошли до текущей промежуточной цели — берём следующую.
            current_goal = WAYPOINTS[waypoint_idx]
            dist_to_goal = float(np.linalg.norm(base_xy - current_goal))

            finished = False

            if dist_to_goal < reach_dist:
                if waypoint_idx < len(WAYPOINTS) - 1:
                    waypoint_idx += 1
                    dwa.reset()

                    current_goal = WAYPOINTS[waypoint_idx]
                    dist_to_goal = float(np.linalg.norm(base_xy - current_goal))

                    print(f"Next waypoint: {waypoint_idx + 1}/{len(WAYPOINTS)}")
                else:
                    finished = True

            if finished and not plot_done:
                plot_navigation_result(
                    trajectory_xy=trajectory_xy,
                    waypoints=WAYPOINTS,
                    wall_rects=WALL_RECTS,
                    dynamic_obstacles=DYNAMIC_OBSTACLES,
                    obstacle_traces=obstacle_traces,
                    goal_xy=WAYPOINTS[-1],
                    save_path="navigation_path.png",
                )
                plot_done = True

            # 6. DWA двигает НЕ робота напрямую, а mocap target.
            if key_callback.run and not key_callback.pause and not finished:
                control_timer += dt

                # Пересчитываем DWA не каждый физический шаг, а примерно раз в 0.08 сек.
                if control_timer >= dwa_cfg.dt:
                    control_timer = 0.0

                    state = np.array([base_x, base_y, base_yaw], dtype=float)

                    current_cmd, traj, info = dwa.plan(
                        state=state,
                        goal_xy=current_goal,
                        circle_obstacles=circle_obstacles,
                        rect_obstacles=WALL_RECTS,
                    )
                    # Emergency rollback:
                    # откатываемся назад ТОЛЬКО если препятствие реально впереди по направлению к цели.
                    # Если оно уже сбоку или позади — назад не едем, чтобы не врезаться в него после объезда.

                    robot_xy = np.array([base_x, base_y], dtype=float)

                    goal_vec = current_goal - robot_xy
                    goal_dist = float(np.linalg.norm(goal_vec))

                    if goal_dist > 1e-6:
                        goal_dir = goal_vec / goal_dist
                    else:
                        goal_dir = np.zeros(2, dtype=float)

                    dwa_clearance = float(info.get("min_clearance", 999.0))

                    emergency_detected = False

                    # Emergency включаем только когда DWA уже видит реально маленький зазор.
                    if dwa_clearance < 0.10:
                        for obs in circle_obstacles:
                            if len(obs) < 5:
                                continue

                            ox, oy, radius, ovx, ovy = obs

                            obs_xy = np.array([ox, oy], dtype=float)
                            obs_vel = np.array([ovx, ovy], dtype=float)

                            robot_to_obs = obs_xy - robot_xy
                            obs_to_robot = robot_xy - obs_xy

                            dist = float(np.linalg.norm(obs_to_robot))
                            if dist < 1e-6:
                                continue

                            clearance = dist - radius - dwa_cfg.robot_radius - dwa_cfg.safety_margin

                            # projection показывает, где препятствие относительно движения к цели:
                            # projection > 0  => препятствие впереди
                            # projection < 0  => препятствие уже позади
                            projection = float(np.dot(robot_to_obs, goal_dir))

                            # lateral показывает боковое смещение препятствия от линии движения.
                            lateral_vec = robot_to_obs - projection * goal_dir
                            lateral = float(np.linalg.norm(lateral_vec))

                            # Если препятствие уже не впереди — НЕ откатываемся назад.
                            if projection <= 0.15:
                                continue

                            # Если препятствие далеко впереди — тоже не emergency, пусть DWA сам объезжает.
                            if projection > 0.85:
                                continue

                            # Если оно сильно сбоку — тоже не rollback.
                            if lateral > 0.55:
                                continue

                            dir_from_obs_to_robot = obs_to_robot / dist
                            closing_speed = float(np.dot(obs_vel, dir_from_obs_to_robot))

                            # Emergency только если цилиндр реально движется на робота.
                            if clearance < 0.16 and closing_speed > 0.06:
                                emergency_detected = True
                                break

                    if emergency_detected and goal_dist > 1e-6:
                        rollback_speed = 0.12
                        rollback_dir = -goal_dir

                        current_cmd = np.array(
                            [
                                rollback_dir[0] * rollback_speed,
                                rollback_dir[1] * rollback_speed,
                                0.0,
                            ],
                            dtype=float,
                        )

                        print("EMERGENCY ROLLBACK", np.round(current_cmd, 3))



                    print(
                        "cmd:",
                        np.round(current_cmd, 3),
                        "clearance:",
                        round(float(info.get("min_clearance", -1.0)), 3),
                        "goal:",
                        round(float(info.get("goal", -1.0)), 3),
                    )

                # vx, vy, omega = current_cmd

                if emergency_timer > 0.0:
                    emergency_timer -= dt
                    cmd_to_apply = emergency_cmd
                else:
                    cmd_to_apply = current_cmd

                vx, vy, omega = cmd_to_apply

                # Двигаем именно НАКОПЛЕННУЮ позицию target,
                # а не каждый раз base_x + маленький шаг.
                desired_mocap_pos[0] += float(vx) * dt
                desired_mocap_pos[1] += float(vy) * dt
                desired_mocap_pos[2] = 0.75

                # Но не даём mocap target улететь слишком далеко от базы.
                base_to_target = desired_mocap_pos[:2] - base_xy
                dist_target = float(np.linalg.norm(base_to_target))

                max_target_ahead = 0.35

                if dist_target > max_target_ahead:
                    desired_mocap_pos[:2] = (
                        base_xy + base_to_target / (dist_target + 1e-9) * max_target_ahead
                    )

                data.mocap_pos[mocap_id] = desired_mocap_pos
                data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0], dtype=float)

            else:
                # Когда остановлено — target держим около текущей базы,
                # чтобы после ENTER он не стартовал из старой позиции.
                desired_mocap_pos = data.mocap_pos[mocap_id].copy()
                control_timer = 0.0
                current_cmd[:] = 0.0
                dwa.reset()

            # 7. Передаём mocap target в mink.
            T_wt = mink.SE3.from_mocap_name(
                model,
                data,
                "pinch_site_target",
            )

            end_effector_task.set_target(T_wt)

            # 8. IK через mink.
            configuration.update(data.qpos)

            for _ in range(max_iters):
                vel = mink.solve_ik(
                    configuration,
                    tasks,
                    rate.dt,
                    solver,
                    damping=1e-3,
                    limits=limits,
                )

                configuration.integrate_inplace(vel, rate.dt)

                err = end_effector_task.compute_error(configuration)
                pos_achieved = bool(np.linalg.norm(err[:3]) <= pos_threshold)
                ori_achieved = bool(np.linalg.norm(err[3:]) <= ori_threshold)

                if pos_achieved and ori_achieved:
                    break

            # 9. Отправляем найденную конфигурацию в MuJoCo.
            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            viewer.sync()
            rate.sleep()


if __name__ == "__main__":
    main()