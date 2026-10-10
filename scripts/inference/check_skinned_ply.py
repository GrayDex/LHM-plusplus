#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Function      : Check a --with_skinning PLY against LHM++'s own posing of one SMPL-X frame

"""
Runs the model once: writes the ``--with_skinning`` export, poses the same Gaussians with
LHM++ itself (``animate_gs_model``, expression zeroed: the export carries none), then poses the
written PLY by plain LBS in NumPy and prints position (mm) and rotation (deg) errors.

Usage:
    python scripts/inference/check_skinned_ply.py \\
        --model_name LHMPP-700M-PixelShuffle \\
        --image_glob "./assets/example_multi_images/00000_yuliang_*.png" \\
        --pose_json ./motion_video/TaiChi/smplx_params/00060.json
"""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

_LHM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _LHM_ROOT)

from plyfile import PlyData
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms.rotation_conversions import quaternion_multiply


def _load_to_gs_ply():
    path = os.path.join(_LHM_ROOT, "scripts/inference/to_gs_ply.py")
    spec = importlib.util.spec_from_file_location("_lhmpp_to_gs_ply", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POSE_KEYS = ("root_pose", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "lhand_pose", "rhand_pose")


def rodrigues(aa: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(aa)
    if angle < 1e-12:
        return np.eye(3)
    k = aa / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def forward_kinematics(aa: np.ndarray, joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Per-joint transforms relative to the zero pose: what LBS about the zero pose applies."""
    world = np.zeros((len(parents), 4, 4))
    for i, p in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = rodrigues(aa[i])
        local[:3, 3] = joints[i] - (joints[p] if p >= 0 else 0.0)
        world[i] = local if p < 0 else world[p] @ local
    rel = world.copy()
    rel[:, :3, 3] -= np.einsum("nij,nj->ni", world[:, :3, :3], joints)
    return rel


def pose_skinned_ply(path: str, aa: np.ndarray, trans: np.ndarray):
    ply = PlyData.read(path)
    v, jt = ply["vertex"], ply["joint"]
    names = v.data.dtype.names
    k = sum(1 for n in names if n.startswith("joint_"))
    x0 = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    q0 = np.stack([v[f"rot_{i}"] for i in range(4)], -1).astype(np.float64)
    idx = np.stack([v[f"joint_{i}"] for i in range(k)], -1).astype(np.int64)
    w = np.stack([v[f"weight_{i}"] for i in range(k)], -1).astype(np.float64)
    joints = np.stack([jt["x"], jt["y"], jt["z"]], -1).astype(np.float64)
    parents = np.asarray(jt["parent"], dtype=np.int64)

    rel = forward_kinematics(aa, joints, parents)
    m = np.einsum("nk,nkij->nij", w, rel[idx])
    xyz = np.einsum("nij,nj->ni", m[:, :3, :3], x0) + m[:, :3, 3] + trans
    r = torch.nn.functional.normalize(matrix_to_quaternion(torch.from_numpy(m[:, :3, :3])), dim=-1)
    q = quaternion_multiply(r, torch.nn.functional.normalize(torch.from_numpy(q0), dim=-1))
    return xyz, q.numpy(), k


def main() -> None:
    tg = _load_to_gs_ply()
    parser = tg.build_arg_parser()
    parser.add_argument("--pose_json", required=True, help="One SMPL-X frame to pose both ways.")
    args = parser.parse_args()
    args.with_skinning = True
    args.pose_dir = ""
    if args.output is None:
        args.output = tg.default_export_output_path(args)

    os.environ.update(
        {
            "APP_ENABLED": "1",
            "APP_MODEL_NAME": args.model_name,
            "APP_TYPE": "infer.human_lrm_a4o",
            "NUMBA_THREADING_LAYER": "omp",
        }
    )
    model, cfg, ref_imgs, _, motion_seqs, _, device = tg.setup_loaders_and_inputs(args)

    with torch.no_grad():
        inf = tg.infer_gaussians(model, ref_imgs, motion_seqs, device)
        dtype = inf.merged_betas.dtype
        gs_attr = inf.gs_model_list[0]

        smplx_zero = tg.build_zero_pose_smplx_params(
            inf.motion_one, inf.transform_mat_neutral_pose, inf.merged_betas, inf.dev, dtype
        )
        export = tg.build_skinned_export(
            model.renderer, gs_attr, inf.query_points, smplx_zero, args.skinning_top_k
        )
        out = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tg.save_skinned_ply(*export, out)

        frame = tg.slice_motion_seq_to_single_frame(
            tg._build_motion_seq_from_pose_json(args.pose_json, cfg), frame_idx=0
        )
        sp = tg.build_animation_frame_smplx_params(
            frame, inf.transform_mat_neutral_pose, inf.merged_betas, inf.dev, dtype
        )
        sp["expr"] = torch.zeros_like(sp["expr"])
        view = model.renderer.get_single_view_smpl_data(sp, 0)
        one = model.renderer._get_single_batch_data(view, 0)
        ref, _ = model.renderer.animate_gs_model(
            gs_attr,
            inf.query_points["neutral_coords"][0],
            one,
            debug=False,
            mesh_meta=inf.query_points["mesh_meta"],
        )
        ref_xyz = ref[0].xyz.double().cpu().numpy()
        ref_q = torch.nn.functional.normalize(ref[0].rotation.double().cpu(), dim=-1).numpy()

    aa = np.concatenate([sp[key].double().cpu().numpy().reshape(-1, 3) for key in POSE_KEYS])
    trans = sp["trans"].double().cpu().numpy().reshape(3)
    xyz, q, k = pose_skinned_ply(out, aa, trans)

    d = np.linalg.norm(xyz - ref_xyz, axis=-1) * 1000.0
    ang = np.degrees(2 * np.arccos(np.clip(np.abs((q * ref_q).sum(-1)), 0.0, 1.0)))
    print(f"{out}: {len(d)} Gaussians, top-{k} joints, frame {args.pose_json}")
    print(f"position mm: mean {d.mean():.3f}  p99 {np.percentile(d, 99):.3f}  max {d.max():.3f}")
    print(f"rotation deg: mean {ang.mean():.3f}  p99 {np.percentile(ang, 99):.3f}  max {ang.max():.3f}")


if __name__ == "__main__":
    main()
