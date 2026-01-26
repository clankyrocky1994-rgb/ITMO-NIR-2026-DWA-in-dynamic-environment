# cycle_adapter.py
import numpy as np

class CycleAdapter:
    def __init__(
        self,
        *,
        model,
        data,
        planner,
        yaw_aligner=None,
        mocap_body="pinch_site_target",
        fingers_act_name="fingers_actuator",
        key_callback=None,
    ):
        self.model = model
        self.data = data
        self.planner = planner
        self.yaw_aligner = yaw_aligner
        self.key_callback = key_callback

        self.path_xy = []
        self.path_idx = 0

        self.mocap_id = model.body(mocap_body).mocapid[0]
        self.fingers_act = model.actuator(fingers_act_name).id

    # ---------- path ----------
    def clear_path(self):
        self.path_xy = []
        self.path_idx = 0

    def plan_to(self, goal_xy):
        if self.path_idx < len(self.path_xy):
            return True

        planned = self.planner.plan(
            self.data.qpos[:2].copy(),
            np.array(goal_xy, dtype=float)
        )
        if planned is None:
            print("❌ A*: path not found")
            self.clear_path()
            return False

        self.path_xy = self.planner.simplify_path(planned, step=2)
        self.path_idx = 0

        if self.yaw_aligner:
            self.yaw_aligner.update_from_path(self.path_xy)

        print(f"✅ New path: {len(self.path_xy)} waypoints")
        return True

    def arrived(self) -> bool:
        return len(self.path_xy) > 0 and self.path_idx >= len(self.path_xy)

    # ---------- mocap ----------
    def set_mocap_xyz(self, xyz):
        self.data.mocap_pos[self.mocap_id] = np.asarray(xyz, dtype=float)
        self.data.mocap_quat[self.mocap_id] = np.array([0, 1, 0, 0], dtype=float)

    # ---------- base ----------
    def set_fix_base(self, v: bool):
        if self.key_callback is not None:
            self.key_callback.fix_base = bool(v)

    # ---------- gripper ----------
    def set_gripper(self, v: int):
        self.data.ctrl[self.fingers_act] = v
