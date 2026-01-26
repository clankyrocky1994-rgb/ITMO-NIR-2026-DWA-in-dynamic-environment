from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np

Mode = Literal[
    "IDLE",
    "NAV_TO_PICK",
    "PREGRASP",
    "GRASP",
    "LIFT",
    "NAV_TO_PLACE",
    "PREPLACE",
    "PLACE",
]

@dataclass
class PPParamsStatic:
    pick_xy: np.ndarray
    place_xy: np.ndarray

    pick_z_pre: float = 0.65
    pick_z_grasp: float = 0.53
    pick_z_lift: float = 0.70
    place_z_pre: float = 0.70
    place_z_drop: float = 0.60

    grip_open: float = 0.0
    grip_close: float = 240.0
    hold_close_steps: int = 120
    hold_open_steps: int = 60

    # dataclass can't have mutable default (np.ndarray) -> use default_factory
    quat: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0, 0], dtype=float))

@dataclass
class Command:
    use_waypoints: bool
    fix_base: bool
    mocap_xyz: Optional[np.ndarray] = None
    mocap_quat: Optional[np.ndarray] = None
    gripper_ctrl: Optional[float] = None
    need_plan: Optional[Literal["pick", "place"]] = None

class PickPlaceStaticLogic:
    def __init__(self, p: PPParamsStatic):
        self.p = p
        self.mode: Mode = "IDLE"
        self.hold = 0

    def start(self) -> None:
        if self.mode == "IDLE":
            self.mode = "NAV_TO_PICK"
            self.hold = 0

    def step(self, path_done: bool, pos_achieved: bool) -> Command:
        if self.mode == "NAV_TO_PICK":
            if path_done:
                self.mode = "PREGRASP"
            return Command(use_waypoints=True, fix_base=False, need_plan="pick")

        if self.mode == "PREGRASP":
            if pos_achieved:
                self.mode = "GRASP"
                self.hold = 0
            target = np.array([self.p.pick_xy[0], self.p.pick_xy[1], self.p.pick_z_pre], float)
            return Command(False, True, target, self.p.quat)

        if self.mode == "GRASP":
            self.hold += 1
            if self.hold >= self.p.hold_close_steps:
                self.mode = "LIFT"
            target = np.array([self.p.pick_xy[0], self.p.pick_xy[1], self.p.pick_z_grasp], float)
            return Command(False, True, target, self.p.quat, gripper_ctrl=self.p.grip_close)

        if self.mode == "LIFT":
            if pos_achieved:
                self.mode = "NAV_TO_PLACE"
                self.hold = 0
            target = np.array([self.p.pick_xy[0], self.p.pick_xy[1], self.p.pick_z_lift], float)
            return Command(False, True, target, self.p.quat)

        if self.mode == "NAV_TO_PLACE":
            if path_done:
                self.mode = "PREPLACE"
            return Command(use_waypoints=True, fix_base=False, need_plan="place")

        if self.mode == "PREPLACE":
            if pos_achieved:
                self.mode = "PLACE"
                self.hold = 0
            target = np.array([self.p.place_xy[0], self.p.place_xy[1], self.p.place_z_pre], float)
            return Command(False, True, target, self.p.quat)

        if self.mode == "PLACE":
            self.hold += 1
            if self.hold >= self.p.hold_open_steps:
                self.mode = "IDLE"
            target = np.array([self.p.place_xy[0], self.p.place_xy[1], self.p.place_z_drop], float)
            return Command(False, True, target, self.p.quat, gripper_ctrl=self.p.grip_open)

        return Command(use_waypoints=False, fix_base=False)
