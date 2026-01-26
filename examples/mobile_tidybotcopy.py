from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

from Astar import AStarGridPlanner

from YawPath import YawAligner

from enum import Enum, auto

class CycleState(Enum):
    IDLE = auto()
    NAV_TO_PICK = auto()
    ARM_PREPICK = auto()      # <-- новое (подойти сверху)
    ARM_DOWN_PICK = auto()
    CLOSE = auto()
    ARM_UP = auto()
    NAV_TO_PLACE = auto()
    ARM_PREPLACE = auto()     # <-- новое (подойти сверху)
    ARM_DOWN_PLACE = auto()
    OPEN = auto()
    DONE = auto()

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"

GOAL1 = np.array([2.55, -4.5, 0.62])   # тумбочка-лоток справа (Z подстрой)
GOAL2 = np.array([-2.55, 6, 0.62])   # тумбочка слева (Z подстрой)
PICK_Z = float() # опускание в лоток 1
PLACE_Z = float()# опускание в лоток 2


@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False
    goal: int = 0
    cycle: bool = False   # <-- добавили

    def __call__(self, key: int) -> None:
        if key == keycodes.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == keycodes.KEY_SPACE:
            self.pause = not self.pause
        elif key == keycodes.KEY_1:
            self.goal = 1
        elif key == keycodes.KEY_2:
            self.goal = 2
        elif key == keycodes.KEY_3:
            self.goal = 3
        elif key == keycodes.KEY_4:
            self.cycle = True


planner = AStarGridPlanner(
    x_min=-3.0, x_max=3.0,
    y_min=-7.0, y_max=7.0,
    resolution=0.25,   
    inflate=0.25,      
)


# Описание препятсвтвий

shelf_hx, shelf_hy = 0.25, 0.90 
shelf_xs = [-0.85, 0.85]
shelf_ys = [-4.2, -1.4, 1.4, 4.2]

for sx in shelf_xs:
    for sy in shelf_ys:
        planner.add_rect_obstacle(cx=sx, cy=sy, hx=shelf_hx, hy=shelf_hy)




planner.build_grid()

path_xy = []
path_idx = 0
Path_Yawning = 100
yaw_aligner = YawAligner(start_on_last_k=Path_Yawning)

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Joints we wish to control.
    # fmt: off
    joint_names = [
        # Base joints.
        "joint_x", "joint_y", "joint_th",
        # Arm joints.
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
    ]
    # fmt: on
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    joint_th_act = model.actuator("joint_th").id

    fingers_id = model.actuator("fingers_actuator").id
    GRIP_OPEN = 0
    GRIP_CLOSED = 255

    cycle_state = CycleState.IDLE
    hold_counter = 0
    
    Z_CARRY = .6          # высота, на которой ты сейчас возишь кубик
    PICK_Z = float(GOAL1[2]) # опускание в лоток 1
    PLACE_Z = float(GOAL2[2])# опускание в лоток 2
    


    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:3] = 100
    damping_task = mink.DampingTask(model, immobile_base_cost)

    tasks = [
        end_effector_task,
        posture_task,
    ]

    limits = [
        mink.ConfigurationLimit(model),
    ]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=True,
        show_right_ui=True,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")
        # reset позы робота
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)

        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # поставить mocap ровно на хват
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        # ещё раз "протащить" кинематику после изменения mocap
        mujoco.mj_forward(model, data)

        pick_xy = None
        place_xy = None
        pick_xy = [2.55, -3.8]
        stoppick = [GOAL1.copy]
        Z_PRE = 0.75      # высота "подойти сверху"
        Z_CARRY = 0.6  # как у тебя

        # и сразу зафиксировать цель IK по этому mocap
        T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
        end_effector_task.set_target(T_wt)
        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)
            # ========== A* -> waypoints -> smooth mocap follow ==========
            mocap_id = model.body("pinch_site_target").mocapid[0]

            if key_callback.cycle and cycle_state in (CycleState.IDLE, CycleState.DONE):
                cycle_state = CycleState.NAV_TO_PICK
                hold_counter = 0
                key_callback.cycle = False
                
                key_callback.goal = 1

            # ====== FSM core (поверх твоей навигации) ======
            # 1) Навигация к лотку 1: ждём пока путь закончится

            if cycle_state in (CycleState.ARM_PREPICK, CycleState.ARM_DOWN_PICK, CycleState.CLOSE,
                            CycleState.ARM_UP, CycleState.ARM_PREPLACE, CycleState.ARM_DOWN_PLACE, CycleState.OPEN):
                key_callback.fix_base = True
            else:
                key_callback.fix_base = False

            if cycle_state == CycleState.NAV_TO_PICK:
                data.ctrl[fingers_id] = GRIP_OPEN
                if path_idx >= len(path_xy) and len(path_xy) > 0:
                    # фиксируем XY там, где реально остановились
                    cur = data.mocap_pos[mocap_id].copy()
                    cycle_state = CycleState.ARM_PREPICK

            
            elif cycle_state == CycleState.ARM_PREPICK:
                data.ctrl[fingers_id] = GRIP_OPEN
                cur = data.mocap_pos[mocap_id].copy()
                target = np.array([2.45, -4.63, Z_PRE], dtype=float)  # подойти сверху
                delta = target - cur
                dist = float(np.linalg.norm(delta))
                step = 0.45 * float(model.opt.timestep)

                if dist < 0.03:
                    data.mocap_pos[mocap_id] = target
                    cycle_state = CycleState.ARM_DOWN_PICK
                else:
                    data.mocap_pos[mocap_id] = cur + (delta / (dist + 1e-9)) * min(step, dist)


            # 2) Опускаем mocap на PICK_Z (рука тянется вниз)
            elif cycle_state == CycleState.ARM_DOWN_PICK:
                data.ctrl[fingers_id] = GRIP_OPEN
                cur = data.mocap_pos[mocap_id].copy()
                target = np.array([pick_xy[0], pick_xy[1], 0.44], dtype=float)  # только Z вниз
                delta = target - cur
                dist = float(np.linalg.norm(delta))
                step = 0.30 * float(model.opt.timestep)

                if dist < 0.02:
                    data.mocap_pos[mocap_id] = target
                    cycle_state = CycleState.CLOSE
                    hold_counter = 0
                else:
                    data.mocap_pos[mocap_id] = cur + (delta / (dist + 1e-9)) * min(step, dist)


            # 3) Закрываем пальцы и держим
            elif cycle_state == CycleState.CLOSE:
                data.ctrl[fingers_id] = GRIP_CLOSED
                hold_counter += 1
                if hold_counter > 60:     # ~0.3 сек на 200 Гц
                    cycle_state = CycleState.ARM_UP

            # 4) Поднимаем mocap обратно на Z_CARRY
            elif cycle_state == CycleState.ARM_UP:
                data.ctrl[fingers_id] = GRIP_CLOSED
                cur = data.mocap_pos[mocap_id].copy()
                target = np.array([GOAL1[0], GOAL1[1], Z_CARRY], dtype=float)
                delta = target - cur
                dist = float(np.linalg.norm(delta))
                step = 0.45 * float(model.opt.timestep)

                if dist < 0.03:
                    data.mocap_pos[mocap_id] = target
                    cycle_state = CycleState.NAV_TO_PLACE
                    # запускаем твой A* к лотку 2:
                    key_callback.goal = 2
                else:
                    data.mocap_pos[mocap_id] = cur + (delta / (dist + 1e-9)) * min(step, dist)

            # 5) Навигация к лотку 2: ждём пока путь закончится
            elif cycle_state == CycleState.NAV_TO_PLACE:
                data.ctrl[fingers_id] = GRIP_CLOSED
                if path_idx >= len(path_xy) and len(path_xy) > 0:
                    cur = data.mocap_pos[mocap_id].copy()
                    cycle_state = CycleState.ARM_PREPLACE

            elif cycle_state == CycleState.ARM_PREPLACE:
                data.ctrl[fingers_id] = GRIP_CLOSED
                cur = data.mocap_pos[mocap_id].copy()
                target = np.array([-2.55, 6.2, 0.6], dtype=float)
                delta = target - cur
                dist = float(np.linalg.norm(delta))
                step = 0.45 * float(model.opt.timestep)

                if dist < 0.03:
                    data.mocap_pos[mocap_id] = target
                    cycle_state = CycleState.ARM_DOWN_PLACE
                else:
                    data.mocap_pos[mocap_id] = cur + (delta / (dist + 1e-9)) * min(step, dist)


            # 6) Опускаем mocap на PLACE_Z
            elif cycle_state == CycleState.ARM_DOWN_PLACE:
                data.ctrl[fingers_id] = GRIP_CLOSED
                cur = data.mocap_pos[mocap_id].copy()
                target = np.array([-2.25, 6.0, 0.565], dtype=float)
                delta = target - cur
                dist = float(np.linalg.norm(delta))
                step = 0.30 * float(model.opt.timestep)

                if dist < 0.02:
                    data.mocap_pos[mocap_id] = target
                    cycle_state = CycleState.OPEN
                    hold_counter = 0
                else:
                    data.mocap_pos[mocap_id] = cur + (delta / (dist + 1e-9)) * min(step, dist)
            # 7) Открываем пальцы
            elif cycle_state == CycleState.OPEN:
                data.ctrl[fingers_id] = GRIP_OPEN
                hold_counter += 1
                if hold_counter > 40:
                    cycle_state = CycleState.DONE

            # 1) если нажали кнопку — строим новый путь
            if key_callback.goal in (1, 2):
                start_xy = data.qpos[:2].copy()
                goal_xy = (GOAL1[:2] if key_callback.goal == 1 else GOAL2[:2]).copy()

                planned = planner.plan(start_xy, goal_xy)
                if planned is None:
                    print("❌ A*: path not found")
                    path_xy = []
                    path_idx = 0
                else:
                    path_xy = planner.simplify_path(planned, step=2)
                    path_idx = 0
                    yaw_aligner.update_from_path(path_xy)
                    print(f"✅ New path: {len(path_xy)} waypoints")

                key_callback.goal = 0

            # 2) если путь есть — двигаем mocap маленькими шагами к текущей точке
            if path_idx < len(path_xy):
                wp = path_xy[path_idx]
                cur = data.mocap_pos[mocap_id].copy()

                target = np.array([wp[0], wp[1], 0.9], dtype=float)  # высота кубика
                delta = target - cur
                dist = float(np.linalg.norm(delta))

                mocap_speed = 2  # м/с (плавность)
                dt = float(model.opt.timestep)
                step = mocap_speed * dt

                if dist < 0.20:
                    # дошли до waypoint -> следующий
                    path_idx += 1
                else:
                    # двигаем кубик в сторону waypoint
                    cur = cur + (delta / (dist + 1e-9)) * min(step, dist)
                    data.mocap_pos[mocap_id] = cur
                    data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0], dtype=float)
# ============================================================


            # Compute velocity and integrate into the next configuration.
            configuration.update(data.qpos)
           
            for i in range(max_iters):
                if key_callback.fix_base:
                    vel = mink.solve_ik(
                        configuration,
                        [*tasks, damping_task],
                        rate.dt,
                        solver,
                        damping=1e-3,
                    )
                else:
                    vel = mink.solve_ik(
                        configuration, tasks, rate.dt, solver, damping=1e-3
                    )
                configuration.integrate_inplace(vel, rate.dt)

                # Exit condition.
                err = end_effector_task.compute_error(configuration)
                pos_achieved = bool(np.linalg.norm(err[:3]) <= pos_threshold)
                ori_achieved = bool(np.linalg.norm(err[3:]) <= ori_threshold)
                if pos_achieved and ori_achieved:
                    break

                if not key_callback.pause:
                    data.ctrl[actuator_ids] = configuration.q[dof_ids]

                    # 🔥 поверх mink — задаём yaw базы напрямую
                    if yaw_aligner.should_apply(path_idx, len(path_xy)) and yaw_aligner.desired_yaw is not None:
                        data.ctrl[joint_th_act] = yaw_aligner.desired_yaw

                    mujoco.mj_step(model, data)
                if yaw_aligner.should_apply(path_idx, len(path_xy)):
                    print("apply yaw:", yaw_aligner.desired_yaw, " current th:", float(data.qpos[2]))
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()