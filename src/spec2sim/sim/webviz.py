"""Browser 3D replay: export what three.js needs to redraw a run, instead of server-rendered video.

Physics is cheap and software rendering is not (~40-180 ms per frame on a CPU server). So the
server records only body poses per frame and the browser draws the scene with three.js:

- `export_meshes(model, out_dir)` writes the robot's visual meshes once (`meshes.bin` + `meshes.json`):
  positions quantized to int16 per mesh, smooth normals as int8, one triangle soup per mesh.
- `Trajectory` records, at video frame rate, the position + quaternion of every body that
  carries a visual geom, then writes `viz.json` (geoms, colors, events, goals) + `traj.bin`
  (float32, frames x bodies x 7).
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from spec2sim.sim.scene import OBJ_PREFIX

VISUAL_GROUP = 2  # Menagerie puts the robot's visual meshes in group 2 (collision meshes in 3)
_TYPES = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
          mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}


def _visual_geoms(m: mujoco.MjModel) -> list[int]:
    out = []
    for g in range(m.ngeom):
        body = m.body(m.geom_bodyid[g]).name
        if (m.geom_group[g] == VISUAL_GROUP or body.startswith(OBJ_PREFIX)) and int(m.geom_type[g]) in _TYPES:
            out.append(g)
    return out


def _rgba(m: mujoco.MjModel, g: int) -> list[float]:
    mat = m.geom_matid[g]
    c = m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g]
    return [round(float(v), 3) for v in c]


def export_meshes(m: mujoco.MjModel, out_dir: Path) -> None:
    """Write every visual mesh as a compact triangle soup. Robot meshes are identical for every
    run, so this runs once at server start and the browser caches the result."""
    out_dir.mkdir(parents=True, exist_ok=True)
    blob, manifest = bytearray(), {}
    used = {int(m.geom_dataid[g]) for g in _visual_geoms(m) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    for mid in sorted(used):
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        na, nn = m.mesh_normaladr[mid], m.mesh_normalnum[mid]
        verts = m.mesh_vert[va:va + vn]
        normals = m.mesh_normal[na:na + nn]
        faces = m.mesh_face[fa:fa + fn]  # vertex indices (relative to the mesh)
        fnorm = m.mesh_facenormal[fa:fa + fn]  # normal indices (relative to the mesh)
        pos = verts[faces.reshape(-1)]
        nrm = normals[fnorm.reshape(-1)]
        lo, hi = pos.min(0), pos.max(0)
        span = np.maximum(hi - lo, 1e-9)
        q = np.round((pos - lo) / span * 65535 - 32768).astype("<i2")
        n8 = np.round(np.clip(nrm, -1, 1) * 127).astype("i1")
        while len(blob) % 4:
            blob.append(0)
        manifest[m.mesh(mid).name] = {"offset": len(blob), "count": len(pos),
                                      "min": lo.round(6).tolist(), "span": span.round(6).tolist()}
        blob += q.tobytes()
        manifest[m.mesh(mid).name]["noffset"] = len(blob)
        blob += n8.tobytes()
    (out_dir / "meshes.bin").write_bytes(bytes(blob))
    (out_dir / "meshes.json").write_text(json.dumps(manifest))


class Trajectory:
    """Body poses per frame for the browser replay."""

    def __init__(self, m: mujoco.MjModel):
        self.m = m
        self.geoms = _visual_geoms(m)
        self.bodies = sorted({int(m.geom_bodyid[g]) for g in self.geoms})
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def capture(self, d: mujoco.MjData) -> None:
        self.frames.append(np.hstack([d.xpos[self.bodies], d.xquat[self.bodies]]).astype("<f4"))
        self.times.append(round(float(d.time), 3))

    def save(self, out_dir: Path, fps: int, events: list[str], goals: list[dict]) -> str:
        m, index = self.m, {b: i for i, b in enumerate(self.bodies)}
        geoms = []
        for g in self.geoms:
            kind = _TYPES[int(m.geom_type[g])]
            geoms.append({
                "body": index[int(m.geom_bodyid[g])], "type": kind,
                "size": [round(float(v), 5) for v in m.geom_size[g]],
                "pos": [round(float(v), 5) for v in m.geom_pos[g]],
                "quat": [round(float(v), 5) for v in m.geom_quat[g]],
                "rgba": _rgba(m, g),
                "mesh": m.mesh(int(m.geom_dataid[g])).name if kind == "mesh" else None,
                "object": m.body(int(m.geom_bodyid[g])).name.removeprefix(OBJ_PREFIX)
                if m.body(int(m.geom_bodyid[g])).name.startswith(OBJ_PREFIX) else None,
            })
        traj = np.stack(self.frames) if self.frames else np.zeros((0, len(self.bodies), 7), "<f4")
        (out_dir / "traj.bin").write_bytes(traj.astype("<f4").tobytes())
        meta = {"fps": fps, "frames": len(self.frames), "bodies": [m.body(b).name for b in self.bodies],
                "times": self.times, "geoms": geoms, "events": events, "goals": goals}
        (out_dir / "viz.json").write_text(json.dumps(meta))
        return str(out_dir / "viz.json")
