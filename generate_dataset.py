"""Generate a reduced-coordinate Franka--TDCR dataset.

The stored configuration is

    q = [q_panda_1, ..., q_panda_7,
         c_0x, c_0y, c_1x, c_1y, c_2x, c_2y]

Therefore, ``qs`` has shape ``(N, 13)``.  The final six values are the
Clark coordinates of the three TDCR segments.  They are converted to the
nine tendon commands with the same OpenCR-MuJoCo kinematics class used by
the TDCR controller.

The stored task-space value is

    x = [px, py, pz, R00, R01, R10, R11, R20, R21]

where the pose is measured after the tendon-controlled TDCR has been
simulated and settled in MuJoCo.

Important:
    The six Clark coordinates are reduced tendon-command coordinates.  The
    many internal TDCR hinge coordinates in MuJoCo are not independent
    dataset inputs and are not stored in ``qs``.

Example:

    python generate_franka_tdcr_clark_dataset_robust.py \\
        --scene assets/example_three_segment_franka_franka_scene.xml \\
        --num-samples 1024 \\
        --max-bending-angle-deg 45 \\
        --output /path/to/franka_tdcr/franka_tdcr_dataset.npz \\
        --plot-output workspace_scatter.png

For the current scene, begin with 30--45 degrees.  Larger bending angles
can create contact-rich and numerically unstable configurations.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm

from opencr_mujoco.tdcr_kinematics import (
    ThreeTendonThreeSegmentTDCRKinematics,
)


N_PANDA = 7
N_TDCR_SEGMENTS = 3
N_TENDONS_PER_SEGMENT = 3
N_TENDONS = N_TDCR_SEGMENTS * N_TENDONS_PER_SEGMENT
N_CLARK = 2 * N_TDCR_SEGMENTS
N_REDUCED_Q = N_PANDA + N_CLARK
N_X = 9


class InvalidSample(RuntimeError):
    """Raised when a requested configuration cannot be simulated safely."""


def get_joint_name(model, joint_id):
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        int(joint_id),
    )


def get_panda_joints_and_actuators(model):
    """Return Panda IDs in the order panda_joint1,...,panda_joint7."""

    joint_ids = []
    qpos_indices = []
    qvel_indices = []
    actuator_ids = []
    joint_names = []

    for joint_number in range(1, N_PANDA + 1):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"panda_joint{joint_number}",
        )

        # Some standalone Panda scenes use panda0_jointN.
        if actuator_id < 0:
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"panda0_joint{joint_number}",
            )

        if actuator_id < 0:
            raise RuntimeError(
                f"Could not find a Panda actuator for joint "
                f"{joint_number}."
            )

        joint_id = int(model.actuator_trnid[actuator_id, 0])

        if joint_id < 0:
            raise RuntimeError(
                f"Actuator for Panda joint {joint_number} is not attached "
                "to a joint."
            )

        joint_ids.append(joint_id)
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))
        actuator_ids.append(int(actuator_id))
        joint_names.append(get_joint_name(model, joint_id))

    return (
        np.asarray(joint_ids, dtype=int),
        np.asarray(qpos_indices, dtype=int),
        np.asarray(qvel_indices, dtype=int),
        np.asarray(actuator_ids, dtype=int),
        joint_names,
    )


def get_tendon_actuator_ids(model):
    """Return the nine TDCR tendon actuators in segment/tendon order."""

    actuator_ids = []

    for segment_id in range(N_TDCR_SEGMENTS):
        for tendon_id in range(N_TENDONS_PER_SEGMENT):
            actuator_name = f"seg_{segment_id}_ten_{tendon_id}"
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )

            if actuator_id < 0:
                raise RuntimeError(
                    f"Could not find tendon actuator '{actuator_name}'."
                )

            actuator_ids.append(int(actuator_id))

    return np.asarray(actuator_ids, dtype=int)


def get_named_keyframe_id(model, key_name):
    """Return a keyframe ID, or -1 if the keyframe does not exist."""

    for key_id in range(model.nkey):
        current_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_id,
        )

        if current_name == key_name:
            return int(key_id)

    return -1


def get_initial_state(model, data, key_name="pretension"):
    """Read qpos, qvel, actuator state, and ctrl from a keyframe."""

    key_id = get_named_keyframe_id(model, key_name)

    if key_id >= 0:
        qpos = model.key_qpos[key_id].copy()
        qvel = model.key_qvel[key_id].copy()
        ctrl = model.key_ctrl[key_id].copy()

        if model.na > 0:
            act = model.key_act[key_id].copy()
        else:
            act = np.zeros(0, dtype=float)

        print(
            f"Using MuJoCo keyframe '{key_name}' for initialization."
        )

        return qpos, qvel, act, ctrl

    print(
        f"Warning: keyframe '{key_name}' was not found. "
        "Using the current model state and current controls."
    )

    qpos = data.qpos.copy()
    qvel = np.zeros(model.nv, dtype=float)
    act = np.zeros(model.na, dtype=float)
    ctrl = data.ctrl.copy()

    return qpos, qvel, act, ctrl


def reset_data(model, data, qpos, qvel, act, ctrl):
    """Reset all state that can persist between independent samples."""

    data.time = 0.0
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = ctrl

    if model.na > 0:
        data.act[:] = act

    data.qacc[:] = 0.0
    data.qfrc_applied[:] = 0.0
    data.xfrc_applied[:] = 0.0

    mujoco.mj_forward(model, data)


def get_panda_joint_limits(model, joint_ids):
    """Return finite Panda joint limits in the selected joint order."""

    lower = []
    upper = []

    for joint_id in joint_ids:
        if not model.jnt_limited[joint_id]:
            name = get_joint_name(model, joint_id)
            raise ValueError(
                f"Panda joint '{name}' does not have finite limits."
            )

        lo, hi = model.jnt_range[joint_id]
        lower.append(float(lo))
        upper.append(float(hi))

    return np.asarray(lower), np.asarray(upper)


def generate_reduced_sobol_samples(
    sampler,
    n_samples,
    panda_low,
    panda_high,
    clark_radius_mm,
):
    """Generate Sobol samples in a Panda box times three Clark disks.

    For each segment, Clark coordinates are sampled uniformly with respect
    to area:

        radius = radius_max * sqrt(u_radius)
        angle  = 2*pi*u_angle

    This samples the physically bounded disk rather than a square.
    """

    unit_samples = sampler.random(n_samples)

    q_samples = np.empty(
        (n_samples, N_REDUCED_Q),
        dtype=np.float64,
    )

    # Franka portion: a joint-limit box.
    q_samples[:, :N_PANDA] = (
        panda_low[None, :]
        + unit_samples[:, :N_PANDA]
        * (panda_high - panda_low)[None, :]
    )

    # TDCR portion: three independent Clark disks.
    for segment_id in range(N_TDCR_SEGMENTS):
        radius_u = unit_samples[:, N_PANDA + 2 * segment_id]
        angle_u = unit_samples[:, N_PANDA + 2 * segment_id + 1]

        radius = clark_radius_mm * np.sqrt(radius_u)
        angle = 2.0 * np.pi * angle_u

        clark_start = N_PANDA + 2 * segment_id
        q_samples[:, clark_start] = radius * np.cos(angle)
        q_samples[:, clark_start + 1] = radius * np.sin(angle)

    return q_samples


def get_tip_pose(model, data):
    """Return [px, py, pz, R00, R01, R10, R11, R20, R21]."""

    try:
        tip_body = data.body("EE_pos")
    except Exception as exc:
        raise RuntimeError(
            "Could not find body 'EE_pos'. Change get_tip_pose() to the "
            "tip body/site name used by your XML scene."
        ) from exc

    position = tip_body.xpos.copy()
    rotation = tip_body.xmat.reshape(3, 3)

    # Same convention as the evaluation workflow:
    # [R00, R01, R10, R11, R20, R21].
    rotation_6d = rotation[:, :2].reshape(-1)
    pose = np.concatenate([position, rotation_6d])

    if pose.shape != (N_X,) or not np.all(np.isfinite(pose)):
        raise InvalidSample("Tip pose is non-finite or has the wrong shape.")

    return pose


def get_tendon_control_target(
    model,
    neutral_ctrl,
    tendon_actuator_ids,
    kinematics,
    clark_coords,
    reject_control_clipping=True,
):
    """Map six Clark coordinates to nine actuator controls.

    The OpenCR controller convention is

        tendon_ctrl = pretension_ctrl + 0.001 * tendon_delta_mm.

    By default, a sample is rejected if its requested controls exceed an
    actuator control range.  This keeps the stored Clark coordinate and the
    simulated tendon command consistent.
    """

    clark_coords = np.asarray(clark_coords, dtype=float)

    if clark_coords.shape != (N_CLARK,):
        raise InvalidSample(
            f"Expected Clark shape {(N_CLARK,)}, got {clark_coords.shape}."
        )

    tendon_delta_mm = np.asarray(
        kinematics.clark_to_tendons_mm(clark_coords),
        dtype=float,
    )

    if tendon_delta_mm.shape != (N_TENDONS,):
        raise InvalidSample(
            "Clark-to-tendon mapping returned shape "
            f"{tendon_delta_mm.shape}, expected {(N_TENDONS,)}."
        )

    if not np.all(np.isfinite(tendon_delta_mm)):
        raise InvalidSample("Clark-to-tendon mapping returned NaN or Inf.")

    target_ctrl = (
        neutral_ctrl[tendon_actuator_ids]
        + 1e-3 * tendon_delta_mm
    )

    for local_id, actuator_id in enumerate(tendon_actuator_ids):
        if not model.actuator_ctrllimited[actuator_id]:
            continue

        lo, hi = model.actuator_ctrlrange[actuator_id]
        value = target_ctrl[local_id]

        if value < lo or value > hi:
            if reject_control_clipping:
                raise InvalidSample(
                    "Tendon actuator control range exceeded."
                )

            target_ctrl[local_id] = np.clip(value, lo, hi)

    if not np.all(np.isfinite(target_ctrl)):
        raise InvalidSample("Tendon control target is non-finite.")

    return target_ctrl


def enforce_fixed_panda_configuration(
    data,
    q_panda,
    panda_qpos_indices,
    panda_qvel_indices,
):
    """Hold the Panda mounting configuration during TDCR simulation."""

    data.qpos[panda_qpos_indices] = q_panda
    data.qvel[panda_qvel_indices] = 0.0


def state_is_valid(model, data, max_abs_qacc):
    """Return whether the current MuJoCo state is numerically usable."""

    arrays = [
        data.qpos,
        data.qvel,
        data.qacc,
        data.ctrl,
    ]

    if model.na > 0:
        arrays.append(data.act)

    if any(not np.all(np.isfinite(array)) for array in arrays):
        return False

    if data.qacc.size > 0:
        if np.max(np.abs(data.qacc)) > max_abs_qacc:
            return False

    return True


def simulate_reduced_configuration(
    model,
    data,
    q_reduced,
    neutral_qpos,
    neutral_qvel,
    neutral_act,
    neutral_ctrl,
    panda_qpos_indices,
    panda_qvel_indices,
    tendon_actuator_ids,
    kinematics,
    ramp_steps,
    settle_steps,
    max_abs_qacc,
    reject_control_clipping=True,
):
    """Simulate one reduced configuration and return the tip pose."""

    q_reduced = np.asarray(q_reduced, dtype=float)

    if q_reduced.shape != (N_REDUCED_Q,):
        raise InvalidSample(
            f"Expected reduced q shape {(N_REDUCED_Q,)}, "
            f"got {q_reduced.shape}."
        )

    q_panda = q_reduced[:N_PANDA]
    q_clark = q_reduced[N_PANDA:]

    tendon_target_ctrl = get_tendon_control_target(
        model=model,
        neutral_ctrl=neutral_ctrl,
        tendon_actuator_ids=tendon_actuator_ids,
        kinematics=kinematics,
        clark_coords=q_clark,
        reject_control_clipping=reject_control_clipping,
    )

    reset_data(
        model=model,
        data=data,
        qpos=neutral_qpos,
        qvel=neutral_qvel,
        act=neutral_act,
        ctrl=neutral_ctrl,
    )

    total_steps = ramp_steps + settle_steps

    for step_id in range(total_steps):
        if ramp_steps > 0 and step_id < ramp_steps:
            alpha = (step_id + 1) / ramp_steps
        else:
            alpha = 1.0

        data.ctrl[:] = neutral_ctrl
        data.ctrl[tendon_actuator_ids] = (
            neutral_ctrl[tendon_actuator_ids]
            + alpha
            * (
                tendon_target_ctrl
                - neutral_ctrl[tendon_actuator_ids]
            )
        )

        enforce_fixed_panda_configuration(
            data=data,
            q_panda=q_panda,
            panda_qpos_indices=panda_qpos_indices,
            panda_qvel_indices=panda_qvel_indices,
        )

        # The Panda qpos is overwritten before every step, so update all
        # derived body/tendon quantities before integrating the TDCR.
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)

        if not state_is_valid(
            model=model,
            data=data,
            max_abs_qacc=max_abs_qacc,
        ):
            raise InvalidSample(
                "MuJoCo state became non-finite or developed excessive qacc."
            )

    # Read the pose at exactly the requested Panda configuration.
    enforce_fixed_panda_configuration(
        data=data,
        q_panda=q_panda,
        panda_qpos_indices=panda_qpos_indices,
        panda_qvel_indices=panda_qvel_indices,
    )
    mujoco.mj_forward(model, data)

    if not state_is_valid(
        model=model,
        data=data,
        max_abs_qacc=max_abs_qacc,
    ):
        raise InvalidSample("Final MuJoCo state is invalid.")

    return get_tip_pose(model, data)


def plot_workspace(x_dataset, output_path, show_plot=False):
    """Save a 3D scatter plot of the tip-position workspace."""

    position = np.asarray(x_dataset)[:, :3]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        s=8,
        alpha=0.75,
        edgecolors="none",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Franka + TDCR Tip Position Workspace")

    position_min = position.min(axis=0)
    position_max = position.max(axis=0)
    ranges = position_max - position_min
    centers = 0.5 * (position_max + position_min)
    radius = max(0.5 * np.max(ranges), 1e-9)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.view_init(elev=25, azim=45)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)
    print(f"Workspace plot saved to: {output_path}")


def make_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 13D Franka + three-segment TDCR dataset "
            "using Clark coordinates."
        )
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="Path to the MuJoCo Franka-TDCR XML scene.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2**17,
        help="Number of accepted dataset samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sobol scrambling seed.",
    )
    parser.add_argument(
        "--tendon-distance-mm",
        type=float,
        default=4.0,
        help="Distance from the backbone to each tendon in millimetres.",
    )
    parser.add_argument(
        "--max-bending-angle-deg",
        type=float,
        default=45.0,
        help=(
            "Maximum Clark bending angle per segment. "
            "Start with 30--45 for this scene."
        ),
    )
    parser.add_argument(
        "--angle-offset-deg",
        type=float,
        nargs=3,
        default=[0.0, 30.0, 60.0],
        metavar=("SEG0", "SEG1", "SEG2"),
        help="Tendon-pattern angle offsets for the three segments.",
    )
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.05,
        help="Time used to ramp from pretension to the target tendon command.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.20,
        help="Time simulated after the tendon-command ramp.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1024,
        help="Number of Sobol candidates generated between batches.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum attempted candidates. Default is max(10*N, N+1000)."
        ),
    )
    parser.add_argument(
        "--max-abs-qacc",
        type=float,
        default=1e8,
        help="Reject a sample if max(abs(qacc)) exceeds this value.",
    )
    parser.add_argument(
        "--allow-control-clipping",
        action="store_true",
        help=(
            "Clip tendon controls to XML ctrlrange instead of rejecting. "
            "Not recommended because stored Clark q then differs from the "
            "requested tendon command."
        ),
    )
    parser.add_argument(
        "--output",
        default="franka_tdcr_dataset.npz",
        help="Output NPZ path. It stores arrays named qs and xs.",
    )
    parser.add_argument(
        "--plot-output",
        default="workspace_scatter.png",
        help="Output workspace-plot path.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the workspace plot interactively.",
    )

    return parser


def main():
    args = make_argument_parser().parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    if args.tendon_distance_mm <= 0.0:
        raise ValueError("--tendon-distance-mm must be positive.")

    if args.max_bending_angle_deg <= 0.0:
        raise ValueError("--max-bending-angle-deg must be positive.")

    if args.ramp_time < 0.0:
        raise ValueError("--ramp-time cannot be negative.")

    if args.settle_time <= 0.0:
        raise ValueError("--settle-time must be positive.")

    if args.candidate_batch_size <= 0:
        raise ValueError("--candidate-batch-size must be positive.")

    if args.max_abs_qacc <= 0.0:
        raise ValueError("--max-abs-qacc must be positive.")

    if len(args.angle_offset_deg) != N_TDCR_SEGMENTS:
        raise ValueError("Exactly three angle offsets are required.")

    if args.max_bending_angle_deg > 45.0:
        print(
            "Warning: bending angles above 45 degrees may produce "
            "contact or numerical-instability warnings in this scene."
        )

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    (
        panda_joint_ids,
        panda_qpos_indices,
        panda_qvel_indices,
        panda_actuator_ids,
        panda_joint_names,
    ) = get_panda_joints_and_actuators(model)

    panda_low, panda_high = get_panda_joint_limits(
        model,
        panda_joint_ids,
    )

    tendon_actuator_ids = get_tendon_actuator_ids(model)

    angle_offsets_rad = np.deg2rad(
        np.asarray(args.angle_offset_deg, dtype=float)
    )
    max_bending_angle_rad = np.deg2rad(
        args.max_bending_angle_deg
    )

    kinematics = ThreeTendonThreeSegmentTDCRKinematics(
        tendon_distance_mm=args.tendon_distance_mm,
        angle_offset_rad_ccw=angle_offsets_rad,
        max_bending_angle_rad=max_bending_angle_rad,
    )

    clark_radius_mm = (
        args.tendon_distance_mm * max_bending_angle_rad
    )

    neutral_qpos, neutral_qvel, neutral_act, neutral_ctrl = (
        get_initial_state(model, data, key_name="pretension")
    )

    timestep = float(model.opt.timestep)
    ramp_steps = int(np.ceil(args.ramp_time / timestep))
    settle_steps = max(
        1,
        int(np.ceil(args.settle_time / timestep)),
    )

    if args.max_attempts is None:
        max_attempts = max(
            10 * args.num_samples,
            args.num_samples + 1000,
        )
    else:
        max_attempts = args.max_attempts

    if max_attempts < args.num_samples:
        raise ValueError("--max-attempts must be at least --num-samples.")

    output_path = Path(args.output)
    plot_path = Path(args.plot_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Panda joints: {panda_joint_names}")
    print(f"Panda q dimension: {len(panda_qpos_indices)}")
    print(f"Clark q dimension: {N_CLARK}")
    print(f"Total reduced q dimension: {N_REDUCED_Q}")
    print(f"Tendon actuator count: {len(tendon_actuator_ids)}")
    print(f"Clark radius per segment: {clark_radius_mm:.6f} mm")
    print(f"MuJoCo timestep: {timestep:.6g} s")
    print(f"Clark-command ramp steps: {ramp_steps}")
    print(f"MuJoCo settle steps per sample: {settle_steps}")
    print(f"Maximum candidate attempts: {max_attempts}")
    print(f"Generating {args.num_samples} accepted samples...")

    sampler = qmc.Sobol(
        d=N_REDUCED_Q,
        scramble=True,
        seed=args.seed,
    )

    q_dataset = np.empty(
        (args.num_samples, N_REDUCED_Q),
        dtype=np.float64,
    )
    x_dataset = np.empty(
        (args.num_samples, N_X),
        dtype=np.float64,
    )

    accepted = 0
    attempts = 0
    rejected = 0

    progress = tqdm(
        total=args.num_samples,
        desc="Generating reduced-coordinate dataset",
    )

    try:
        while accepted < args.num_samples:
            remaining = args.num_samples - accepted
            batch_size = min(args.candidate_batch_size, remaining)

            candidate_batch = generate_reduced_sobol_samples(
                sampler=sampler,
                n_samples=batch_size,
                panda_low=panda_low,
                panda_high=panda_high,
                clark_radius_mm=clark_radius_mm,
            )

            for q_candidate in candidate_batch:
                attempts += 1

                if attempts > max_attempts:
                    raise RuntimeError(
                        "Maximum candidate attempts exceeded. "
                        f"Accepted {accepted}/{args.num_samples}; "
                        f"rejected {rejected}. Try reducing "
                        "--max-bending-angle-deg or increasing "
                        "--max-attempts."
                    )

                try:
                    x_candidate = simulate_reduced_configuration(
                        model=model,
                        data=data,
                        q_reduced=q_candidate,
                        neutral_qpos=neutral_qpos,
                        neutral_qvel=neutral_qvel,
                        neutral_act=neutral_act,
                        neutral_ctrl=neutral_ctrl,
                        panda_qpos_indices=panda_qpos_indices,
                        panda_qvel_indices=panda_qvel_indices,
                        tendon_actuator_ids=tendon_actuator_ids,
                        kinematics=kinematics,
                        ramp_steps=ramp_steps,
                        settle_steps=settle_steps,
                        max_abs_qacc=args.max_abs_qacc,
                        reject_control_clipping=(
                            not args.allow_control_clipping
                        ),
                    )
                except InvalidSample:
                    rejected += 1
                    continue

                q_dataset[accepted] = q_candidate
                x_dataset[accepted] = x_candidate
                accepted += 1
                progress.update(1)

                if accepted % 100 == 0:
                    progress.set_postfix(
                        attempts=attempts,
                        rejected=rejected,
                    )
    finally:
        progress.close()

    x_max = x_dataset[:, :3].max(axis=0)
    print(
        f"x_max [m] = [{x_max[0]:.6f}, "
        f"{x_max[1]:.6f}, {x_max[2]:.6f}]"
    )
    print(f"Candidate attempts: {attempts}")
    print(f"Rejected candidates: {rejected}")

    # Preserve the existing workflow exactly: only qs and xs are stored.
    np.savez(
        output_path,
        qs=q_dataset,
        xs=x_dataset,
    )

    print(f"Saved {len(q_dataset)} samples to {output_path}")
    print(f"q shape: {q_dataset.shape}")
    print(f"x shape: {x_dataset.shape}")

    plot_workspace(
        x_dataset=x_dataset,
        output_path=plot_path,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
