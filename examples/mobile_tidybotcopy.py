from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink
from mink.contrib.keyboard_teleop import keycodes

# from Astar import AStarGridPlanner

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"

GOAL1 = np.array([2.55, -3.8, 0.62])   # тумбочка-лоток справа (Z подстрой)
GOAL2 = np.array([-2.55, 6.2, 0.62])   # тумбочка слева (Z подстрой)

@dataclass
class KeyCallback:
    fix_base: bool = False
    pause: bool = False
    goal: int = 0  # 0 = ничего, 1/2/3 = пресеты

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

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update task target.
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)
            # --- jump mocap target to preset goals by keyboard ---
            mocap_id = model.body("pinch_site_target").mocapid[0]

            if key_callback.goal == 1:
                data.mocap_pos[mocap_id] = GOAL1
                data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0], dtype=float)
                key_callback.goal = 0  # сброс, чтобы не телепортировало каждый кадр

            elif key_callback.goal == 2:
                data.mocap_pos[mocap_id] = GOAL2
                data.mocap_quat[mocap_id] = np.array([0, 1, 0, 0], dtype=float)
                key_callback.goal = 0


            # Compute velocity and integrate into the next configuration.
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
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
