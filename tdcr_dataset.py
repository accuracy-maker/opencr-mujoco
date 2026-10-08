"""
Generate dataset in Clarke Coordinate for TDCR robot

1. generate the xml model file
2. read the xml model file
3. random genearte the configs
4. only accept configs within tendon limits
5. compute the tip's pose
6. save the (q, x) pairs
"""

import re
import math
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

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


# sample Clarke configs
def sample_clarke(rng, radii_mm):
	q = np.zeros(2 * radii_mm.size, dtype=float)
	for segment, radius in enumerate(radii_mm):
		magnitude = radius * math.sqrt(float(rng.random()))
		angle = 2.0 * math.pi * float(rng.random())
		q[2 * segment : 2 * segment + 2] = magnitude * np.array(
		    [math.cos(angle), math.sin(angle)]
		)
	return q	

def check_control_limits(model, actuator_ids, control, tolerance=1e-10):
	limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
	ranges = np.asarray(model.actuator_ctrlrange, dtype=float)

	for actuator_id in actuator_ids:
		if not limited[actuator_id]:
		    continue
		lower, upper = ranges[actuator_id]
		value = control[actuator_id]
		if value < lower - tolerance or value > upper + tolerance:
	            return False
	return True

def plot_positions(positions):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for sample_id in range(positions.shape[0]):
        sample_positions = positions[sample_id]

        ax.plot(
            sample_positions[:, 0],
            sample_positions[:, 1],
            sample_positions[:, 2],
            "-o",
            linewidth=1.2,
            markersize=2,
            alpha=0.7,
        )

    ax.scatter(
        positions[:, 0, 0],
        positions[:, 0, 1],
        positions[:, 0, 2],
        color="green",
        s=30,
        label="Base",
    )

    ax.scatter(
        positions[:, -1, 0],
        positions[:, -1, 1],
        positions[:, -1, 2],
        color="red",
        s=30,
        label="Tip",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("TDCR Skeletons")
    ax.legend()
    ax.set_box_aspect((1, 1, 1))

    plt.show()

def get_body_ids(model):
    body_ids = []

    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id)
        # print(body_name)
        if body_name:
            body_ids.append(body_id)
    body_ids.sort()
    return body_ids

MAX_BENDING_ANGLE_RAD = np.pi / 3
NUM_SAMPLES = 100000

xml_path = "assets/tdcr/ftdcr_v4_sysid.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

actuator_ids, tendons_per_segment = find_tendon_actuators(model)

n_segments = len(tendons_per_segment)

distances, offsets = infer_geometry(xml_path, n_segments)

if len(distances) == 1:
    distances = list(distances) * n_segments
if len(offsets) == 1:
    offsets = list(offsets) * n_segments
if len(distances) != n_segments or len(offsets) != n_segments:
    raise ValueError(
        "--tendon-distance-mm and --angle-offset-rad must contain either "
        "one value or one value per segment"
    )

distances = np.asarray(distances, dtype=float)
offsets = np.asarray(offsets, dtype=float)
print(f"distances: {distances}")
radii_mm = distances * MAX_BENDING_ANGLE_RAD

print(f"radii_mm: {radii_mm}")

kin = MultiSegmentTDCRKinematics(
    n_tendons_per_segment=tendons_per_segment,
    tendon_distances_mm=distances,
    angle_offsets_rad_ccw=offsets,
    max_bending_angles_rad = MAX_BENDING_ANGLE_RAD
)

key_id, pretension_ctrl = read_pretension_keyframe(model)

rng = np.random.default_rng(42)

settle_steps = max(1, int(round(1.0 / model.opt.timestep)))

tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "EE_pos")

body_ids = get_body_ids(model)

positions = []

for i in tqdm(range(NUM_SAMPLES), desc="generating"):
    q = sample_clarke(rng, radii_mm)
    delta_tendons_m = kin.clark_to_tendons_mm(q) * 1e-3
    requested_ctrl = pretension_ctrl.copy()
    for actuator_id, delta in zip(actuator_ids, delta_tendons_m):
    	requested_ctrl[actuator_id] += delta

    if not check_control_limits(model, actuator_ids, requested_ctrl):
        print(f"sample: {i+1} rejected | reason: out of tendon limits")
        continue

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.ctrl[:] = pretension_ctrl
    for actuator_id, delta in zip(actuator_ids, delta_tendons_m):
    	data.ctrl[actuator_id] = pretension_ctrl[actuator_id] + delta
    mujoco.mj_forward(model, data)

    for _ in range(settle_steps):
    	mujoco.mj_step(model, data)

    positions.append(np.array([data.xpos[body_id] for body_id in body_ids]))

positions = np.asarray(positions)
print(f"positions shape: {positions.shape}")

# plot_positions(positions)
