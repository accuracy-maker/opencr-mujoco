import argparse
import re

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.stats import qmc
from tqdm import tqdm


TDCR_JOINT_PATTERN = re.compile(r"^joint_\d+_.+$")


def get_joint_name(model, joint_id):
    return mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id,
    )


def get_robot_joint_ids(model):
    """Return Franka and TDCR hinge-joint IDs."""

    franka_joint_ids = []

    for i in range(1, 8):
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            f"panda_joint{i}",
        )

        if actuator_id < 0:
            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                f"panda0_joint{i}",
            )

        if actuator_id >= 0:
            joint_id = model.actuator_trnid[actuator_id, 0]
            franka_joint_ids.append(int(joint_id))

    tdcr_joint_ids = []

    for joint_id in range(model.njnt):
        joint_name = get_joint_name(model, joint_id)

        if joint_name is None:
            continue

        is_tdcr_joint = TDCR_JOINT_PATTERN.match(joint_name) is not None
        is_hinge = (
            model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_HINGE
        )

        if is_tdcr_joint and is_hinge:
            tdcr_joint_ids.append(joint_id)

    all_joint_ids = sorted(
        set(franka_joint_ids + tdcr_joint_ids),
        key=lambda jid: model.jnt_qposadr[jid],
    )

    qpos_indices = np.array(
        [model.jnt_qposadr[jid] for jid in all_joint_ids],
        dtype=int,
    )

    joint_names = [
        get_joint_name(model, jid)
        for jid in all_joint_ids
    ]

    return all_joint_ids, qpos_indices, joint_names


def get_joint_sampling_limits(
    model,
    joint_ids,
    tdcr_limit_deg=35.0,
):
    """Get sampling bounds for each robot joint."""

    lower = []
    upper = []

    tdcr_limit = np.deg2rad(tdcr_limit_deg)

    for joint_id in joint_ids:
        joint_name = get_joint_name(model, joint_id)

        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]

        elif joint_name.startswith("joint_"):
            # TDCR joints are often unlimited in the XML.
            lo = -tdcr_limit
            hi = tdcr_limit

        else:
            raise ValueError(
                f"No sampling range available for {joint_name}"
            )

        lower.append(lo)
        upper.append(hi)

    return np.asarray(lower), np.asarray(upper)


def get_tip_pose(model, data):
    """
    Return TDCR tip pose in the world frame.

    Output:
        [px, py, pz, qw, qx, qy, qz]
    """

    tip_body = data.body("EE_pos")

    position = tip_body.xpos.copy()

    # MuJoCo quaternion convention: [w, x, y, z].
    quaternion_wxyz = tip_body.xquat.copy()

    return np.concatenate(
        [position, quaternion_wxyz]
    )


def plot_workspace(
    x_dataset,
    output_path,
    show_plot=False,
):
    """Plot the TDCR tip position workspace."""

    position = x_dataset[:, :3]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        position[:, 0],
        position[:, 1],
        position[:, 2],
        s=20,
        alpha=0.95,
        edgecolors="none",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(
        "Franka + TDCR Tip Position Workspace"
    )

    # Equal aspect ratio.
    position_min = position.min(axis=0)
    position_max = position.max(axis=0)

    ranges = position_max - position_min
    centers = 0.5 * (position_max + position_min)
    radius = 0.5 * np.max(ranges)

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius,
    )
    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius,
    )
    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius,
    )

    ax.view_init(
        elev=25,
        azim=45,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()

    plt.close(fig)

    print(
        f"Workspace plot saved to: {output_path}"
    )


def generate_sobol_samples(
    q_low,
    q_high,
    num_samples,
    seed,
):
    """Generate Sobol samples in the joint-limit box."""

    dimension = len(q_low)

    sampler = qmc.Sobol(
        d=dimension,
        scramble=True,
        seed=seed,
    )

    # Sobol sequences are best balanced for N = 2^m.
    is_power_of_two = (
        num_samples > 0
        and (num_samples & (num_samples - 1)) == 0
    )

    if is_power_of_two:
        m = int(np.log2(num_samples))
        unit_samples = sampler.random_base2(m=m)
    else:
        print(
            "Warning: num_samples is not a power of two. "
            "Using Sobol.random(num_samples)."
        )
        unit_samples = sampler.random(num_samples)

    return q_low + unit_samples * (q_high - q_low)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scene",
        required=True,
        help="Path to the MuJoCo XML scene.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=2**17,
        help="Number of Sobol samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sobol scrambling seed.",
    )

    parser.add_argument(
        "--tdcr-limit-deg",
        type=float,
        default=35.0,
        help="Sampling limit for unlimited TDCR joints.",
    )

    parser.add_argument(
        "--output",
        default="franka_tdcr_dataset.npz",
        help="Output dataset path.",
    )

    parser.add_argument(
        "--plot-output",
        default="workspace_scatter.png",
        help="Output workspace plot path.",
    )

    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the workspace plot interactively.",
    )

    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError(
            "num_samples must be positive."
        )

    model = mujoco.MjModel.from_xml_path(
        args.scene
    )
    data = mujoco.MjData(model)

    joint_ids, qpos_indices, joint_names = (
        get_robot_joint_ids(model)
    )

    if len(qpos_indices) == 0:
        raise RuntimeError(
            "No Franka or TDCR joints were found."
        )

    q_low, q_high = get_joint_sampling_limits(
        model,
        joint_ids,
        tdcr_limit_deg=args.tdcr_limit_deg,
    )

    print(
        f"Number of configuration DoF: "
        f"{len(qpos_indices)}"
    )

    print(
        f"Generating {args.num_samples} Sobol samples..."
    )

    q_samples = generate_sobol_samples(
        q_low=q_low,
        q_high=q_high,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    q_dataset = np.zeros(
        (
            args.num_samples,
            len(qpos_indices),
        ),
        dtype=np.float64,
    )

    # x = [px, py, pz, qw, qx, qy, qz].
    x_dataset = np.zeros(
        (
            args.num_samples,
            7,
        ),
        dtype=np.float64,
    )

    neutral_qpos = data.qpos.copy()

    for sample_id in tqdm(range(args.num_samples), desc="Generating dataset"):
        q_sample = q_samples[sample_id]

        data.qpos[:] = neutral_qpos
        data.qpos[qpos_indices] = q_sample
        data.qvel[:] = 0.0

        # Recompute all body poses from q.
        mujoco.mj_forward(model, data)

        q_dataset[sample_id] = (
            data.qpos[qpos_indices]
        )

        x_dataset[sample_id] = get_tip_pose(
            model,
            data,
        )

    np.savez_compressed(
        args.output,
        q=q_dataset,
        x=x_dataset,
        q_joint_names=np.asarray(
            joint_names
        ),
        pose_convention=np.asarray(
            [
                "px",
                "py",
                "pz",
                "qw",
                "qx",
                "qy",
                "qz",
            ]
        ),
    )

    plot_workspace(
        x_dataset=x_dataset,
        output_path=args.plot_output,
        show_plot=args.show_plot,
    )

    print(f"Dataset saved to: {args.output}")
    print(f"q shape: {q_dataset.shape}")
    print(f"x shape: {x_dataset.shape}")


if __name__ == "__main__":
    main()