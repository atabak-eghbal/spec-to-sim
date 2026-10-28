"""Fixed motion controller: turns validated plan actions into arm joint targets.

- Inverse kinematics: damped least squares on the 7 arm joints, solved on a scratch MjData so
  the physics state is untouched. The gripper always points straight down; `yaw` spins it about z.
- Execution: straight-line Cartesian paths with smoothstep timing, IK every control tick,
  position actuators + gravity compensation.
- Grasp assist (documented in README): when the gripper closes within reach of an object, a weld
  holds it to the hand until the gripper opens. Physical-only grasping proved too flaky to demo.
"""
from __future__ import annotations

import mujoco
import numpy as np

from spec2sim.sim.scene import OBJ_PREFIX, Scene

HOME_Q = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
GRASP_REACH = 0.02  # tcp may be this far outside an object's half-extent and still grasp
MAX_GRASP_WIDTH = 0.078  # Panda fingers open to 8 cm total


class IKError(RuntimeError):
    pass


def _width_across(o, R: np.ndarray, axis: np.ndarray) -> float:
    """Object extent along the finger-closing axis, using its real shape (not its bounding box)."""
    if o.shape == "sphere":
        return max(o.size)
    if o.shape == "cylinder":
        c = abs(float(R[:, 2] @ axis))  # cos(angle) between cylinder axis and finger axis
        return max(o.size[:2]) * np.sqrt(max(0.0, 1 - c * c)) + o.size[2] * c
    return float(np.sum(np.abs(R.T @ axis) * np.array(o.size)))


def _rz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Controller:
    def __init__(self, scene: Scene, data: mujoco.MjData):
        self.s, self.m, self.d = scene, scene.model, data
        self.scratch = mujoco.MjData(self.m)
        self.tcp = self.m.site("tcp").id
        self.hand = self.m.body("hand").id
        jr = self.m.jnt_range[[self.m.joint(f"joint{i}").id for i in range(1, 8)]]
        self.q_lo, self.q_hi = jr[:, 0] + 0.02, jr[:, 1] - 0.02
        # Reference "pointing down" orientation = TCP orientation at the home pose.
        self.scratch.qpos[self.s.arm_qpos] = HOME_Q
        mujoco.mj_kinematics(self.m, self.scratch)
        self.R_home = self.scratch.site_xmat[self.tcp].reshape(3, 3).copy()
        self.q_target = HOME_Q.copy()
        self.grip = GRIP_OPEN
        self.held: str | None = None

    # ---------- kinematics ----------
    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.tcp].copy()

    def ik(self, pos: np.ndarray, yaw: float, q0: np.ndarray, iters: int = 200) -> np.ndarray:
        m, sd = self.m, self.scratch
        R_des = _rz(yaw) @ self.R_home
        q = q0.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            sd.qpos[self.s.arm_qpos] = q
            mujoco.mj_kinematics(m, sd)
            mujoco.mj_comPos(m, sd)
            e_p = pos - sd.site_xpos[self.tcp]
            R = sd.site_xmat[self.tcp].reshape(3, 3)
            e_r = 0.5 * sum(np.cross(R[:, i], R_des[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < 1e-3 and np.linalg.norm(e_r) < 1e-2:
                return q
            mujoco.mj_jacSite(m, sd, jacp, jacr, self.tcp)
            J = np.vstack([jacp[:, self.s.arm_dof], jacr[:, self.s.arm_dof]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            # Small pull toward home keeps the elbow posture natural (null-space bias).
            N = np.eye(7) - np.linalg.pinv(J) @ J
            dq += N @ (0.05 * (HOME_Q - q))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.q_lo, self.q_hi)
        sd.qpos[self.s.arm_qpos] = q
        mujoco.mj_kinematics(m, sd)
        err = np.linalg.norm(pos - sd.site_xpos[self.tcp])
        if err > 0.01:
            raise IKError(f"target {np.round(pos, 3).tolist()} unreachable (IK error {err * 100:.1f} cm)")
        return q

    def object_pos(self, name: str) -> np.ndarray:
        if name not in self.s.objects:
            raise IKError(f"plan refers to unknown object '{name}'")
        return self.d.body(OBJ_PREFIX + name).xpos.copy()

    # ---------- per-tick control ----------
    def apply(self) -> None:
        self.d.ctrl[:7] = self.q_target
        self.d.ctrl[7] = self.grip
        self.d.qfrc_applied[self.s.arm_dof] = self.d.qfrc_bias[self.s.arm_dof]  # gravity comp

    # ---------- grasp assist ----------
    def try_grasp(self) -> tuple[str | None, str]:
        """Attach the nearest graspable object. Returns (name or None, human-readable reason)."""
        tcp = self.tcp_pos()
        finger_axis = self.d.xmat[self.hand].reshape(3, 3)[:, 1]  # fingers slide along hand y
        best, best_d, reason = None, np.inf, ""
        for name, o in self.s.objects.items():
            if o.fixed:
                continue
            b = self.d.body(OBJ_PREFIX + name)
            R = b.xmat.reshape(3, 3)
            local = R.T @ (tcp - b.xpos)  # tcp in object frame
            half = np.array(o.size) / 2
            d = np.linalg.norm(np.maximum(np.abs(local) - half, 0))
            if d > GRASP_REACH or d >= best_d:
                continue
            width = _width_across(o, R, finger_axis)
            if width > MAX_GRASP_WIDTH:
                reason = (f"{name} is {width * 100:.1f} cm wide across the fingers; "
                          f"Panda gripper opens only {MAX_GRASP_WIDTH * 100:.1f} cm")
                continue
            best, best_d = name, d
        if best is not None:
            self._weld(best, True)
            self.held = best
            return best, f"grasped {best}"
        return None, reason or f"gripper closed on nothing (tcp at {np.round(tcp, 3).tolist()})"

    def release(self) -> str | None:
        name, self.held = self.held, None
        if name:
            self._weld(name, False)
        return name

    def _weld(self, name: str, on: bool) -> None:
        eq = self.s.weld_ids[name]
        if on:
            # relpose = object pose in hand frame, so the weld holds it exactly where it is now.
            obj = self.d.body(OBJ_PREFIX + name)
            hp, hR = self.d.xpos[self.hand], self.d.xmat[self.hand].reshape(3, 3)
            hq = self.d.xquat[self.hand]
            rel_p = hR.T @ (obj.xpos - hp)
            hq_inv = np.zeros(4)
            mujoco.mju_negQuat(hq_inv, hq)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, hq_inv, obj.xquat)
            self.m.eq_data[eq, :3] = 0
            self.m.eq_data[eq, 3:6] = rel_p
            self.m.eq_data[eq, 6:10] = rel_q
            self.m.eq_data[eq, 10] = 1.0
        self.d.eq_active[eq] = on
