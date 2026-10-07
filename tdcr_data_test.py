"""
Generate dataset in Clarke Coordinate for TDCR robot

1. generate the xml model file
2. read the xml model file and read the configurations range
3. random genearte the configs within the limits
4. compute the tip's pose
5. save the (q, x) pairs
"""

import re
import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from opencr_mujoco.tdcr_kinematics import MultiSegmentTDCRKinematics

def find_tendon_actuators(model):
    """return tendon actuator IDs in segment-major order"""
    found = {}
    pattern = re.compile(r"^seg_(\d+)_ten_(\d+)$")

    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
        match = pattern.match(name or "")

        if match:
            segment = int(match.group(1))
            tendon = int(match.group(2))
            found.setdefault(segment, {})[tendon] = actuator_id

    if not found:
        raise RuntimeError("No  seg_<segment>_ten_<tendon> actuators found")

    n_segments = max(found) + 1
    actuator_ids = []
    tendons_per_segment = []

    for segment in range(n_segments):
        if segment not in found:
            raise RuntimeError(f"Missing tendon actuator for segment {segment}")

        tendon_ids = found[segment]
        expected = list(range(len(tendon_ids)))

        if sorted(tendon_ids) != expected:
            raise RuntimeError(
                f"Tendon indices for segment {segment} are not contiguous: "
                f"{sorted(tendon_ids)}"
                )

        tendons_per_segment.append(len(tendon_ids))
        actuator_ids.extend(tendon_ids[i] for i in expected)

    return actuator_ids, tendons_per_segment

def infer_geometry(xml_path, n_segments):
    """infer geometry from a generated XML"""
    try:
        from generate import tdcr_geometry_from_scene

        geometry = tdcr_geometry_from_scene(xml_path)
    except Exception:
        geometry = {}

    distances = geometry.get("tendon_distance_mm")
    offsets = geometry.get("angle_offset_rad_ccw")

    if distances is None:
        distances = [4.0] * n_segments
        print(
            "Warning: tendon distance was not inferred; using 4.0 mm. "
            "Pass --tendon-distance-mm explicitly for another robot."
        )
    elif np.isscalar(distances):
        distances = [float(distances)] * n_segments
    else:
        distances = [float(value) for value in distances]

    if offsets is None: 
        offsets = [segment * math.pi / 6.0 for segment in range(n_segments)]
        print(
            "Warning: tendon angle offsets were not inferred; using "
            "[0, pi/6, ...]. Pass --angle-offset-rad explicitly if needed."
        )
    else:
        offsets = [float(value) for value in offsets]

    if len(distances) != n_segments or len(offsets) != n_segments:
        raise ValueError(
            "The number of tendon distances and angle offsets must match "
            f"the detected number of segments ({n_segments})"
        )

    return distances, offsets

# pretension is the initial tensile force applied to every tendon when
# TDCR is in its neutral configuration
# mainly for maintaining the backbone stable
def read_pretension_keyframe(model):
    """return the pretension keyframe ID and its full atuator-control vector"""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "pretension")
    if key_id < 0:
        raise RuntimeError("The XML does not contain a 'pretension' keyframe")
    return key_id, model.key_ctrl[key_id].copy()

xml_path = "assets/tdcr/ftdcr_v4_sysid.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

actuator_ids, tendons_per_segment = find_tendon_actuators(model)

print(f"length of actuator_ids: {len(actuator_ids)}\n"
    f"length of tendons per segment: {len(tendons_per_segment)}")

print(f"actuator_ids:\n {actuator_ids}\n",
    f"tendons per segment:\n {tendons_per_segment}")

n_segments = len(tendons_per_segment)

distances, offsets = infer_geometry(xml_path, n_segments)

print(f"inferred distances: {distances}\n",
    f"inferred offsets: {offsets}")

if len(distances) == 1:
    distances = list(distances) * n_segments
if len(offsets) == 1:
    offsets = list(offsets) * n_segments
if len(distances) != n_segments or len(offsets) != n_segments:
    raise ValueError(
        "--tendon-distance-mm and --angle-offset-rad must contain either "
        "one value or one value per segment"
    )

key_id, pretension_ctrl = read_pretension_keyframe(model)
print(f"key id: {key_id}\n",
    f"pretension_ctrl is:\n {pretension_ctrl}")

kin = MultiSegmentTDCRKinematics(
    n_tendons_per_segment=tendons_per_segment,
    tendon_distances_mm=distances,
    angle_offsets_rad_ccw=offsets,
)

mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

q = np.array([0, 2, 0, 0, 0, 0])

delta_tendon_m = kin.clark_to_tendons_mm(q) * 1e-3
print(f"delta_tendon_m: {delta_tendon_m}")

data.ctrl[:] = pretension_ctrl
for actuator_id, delta in zip(actuator_ids, delta_tendon_m):
    data.ctrl[actuator_id] = pretension_ctrl[actuator_id] + delta

settle_steps = max(1, int(round(1.0 / model.opt.timestep)))
print(f"settle_steps: {settle_steps}")

for _ in range(settle_steps):
    mujoco.mj_step(model, data)

tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")
print(f"tip id: {tip_id}")

position = data.xpos[tip_id].copy()
quaternion = data.xquat[tip_id].copy()
print(f"position:\n {position}\n",
    f"quaternion:\n {quaternion}")

R = Rotation.from_quat(
    [
        quaternion[1],
        quaternion[2],
        quaternion[3],
        quaternion[0],
    ])

print(f"rotation matrix is:\n {R}")