# -*- coding: utf-8 -*-#
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-08 21:42:24, Version 0.0, SMPLX + FLAME2019 + Voxel-Based Queries.
# @Function      : voxel-based-skinning class
# @Description   : 1.canonical query, 2.offset, 3.blendshape -> 4.posed-view

import copy
import os
import sys

sys.path.append("./")

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.io import load_ply, save_ply
from pytorch3d.ops import knn_points
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from smplx.lbs import batch_rigid_transform
from torch.nn import functional as F

from core.models.rendering.mesh_utils import Mesh
from core.models.rendering.skinnings.base_skinning import BaseSkinning
from core.models.rendering.smplx.smplx.lbs import blend_shapes


def avaliable_device():

    import torch

    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"

    return device


class SMPLXVoxelSkinning(nn.Module):
    """
    SMPLXVoxelSkinning

    This class implements voxel-based skinning for SMPL-X and FLAME2019 human models.
    It leverages a dense sampling of surface/body/face points and applies voxel-based
    blend skinning weights for smooth mesh deformations during animated pose transformations.

    Main Functions:
        - Integrates the BaseSkinning module to build canonical (rest) pose shape and joint structures.
        - Supports dense sampling and upsampling of mesh surfaces and their associated features.
        - Provides querying of voxelized skinning weights for arbitrary points in the canonical space.
        - Suitable for neural avatar and implicit model pipelines requiring per-point or per-vertex skinning.

    Usage:
        Construct this class, use `query_voxel_skinning_weights()` for per-point weights,
        or access upsampled meshes and normals for advanced avatar/animation tasks.
    """

    def __init__(
        self,
        human_model_path,
        gender,
        subdivide_num,
        expr_param_dim=50,
        shape_param_dim=100,
        cano_pose_type=0,
        body_face_ratio=3,
        dense_sample_points=40000,
        apply_pose_blendshape=False,
    ):
        """
        Initialize the SMPLXVoxelSkinning module for voxel-based skinning of SMPL-X and FLAME2019 human models.

        Args:
            human_model_path (str): Path to the directory containing SMPL-X and FLAME model files.
            gender (str): Gender specification, one of ['male', 'female', 'neutral'].
            subdivide_num (int): Number of mesh subdivision iterations for upsampling mesh detail.
            expr_param_dim (int, optional): Number of facial expression parameters. Defaults to 50.
            shape_param_dim (int, optional): Number of shape (beta) parameters. Defaults to 100.
            cano_pose_type (int, optional): Canonical pose configuration type. Defaults to 0.
            body_face_ratio (int, optional): Ratio of body versus face samples in dense surface sampling. Defaults to 3.
            dense_sample_points (int, optional): Total number of points to sample from body/face surfaces. Defaults to 40000.
            apply_pose_blendshape (bool, optional): If True, apply pose blendshapes in canonical mesh initialization. Defaults to False.

        Description:
            This constructor sets up the underlying BaseSkinning module, the SMPL-X/FLAME model layer, mesh upsampling,
            and dense sampling points for subsequent per-vertex or per-point voxel weight querying and canonical-to-posed
            mesh transformations.
        """

        super().__init__()

        # define base_skinning
        self.base_skinning = BaseSkinning(
            human_model_path=human_model_path,
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            subdivide_num=subdivide_num,
            cano_pose_type=cano_pose_type,
        )

        self.smplx_layer = copy.deepcopy(self.base_skinning.layer[gender])

        self.apply_pose_blendshape = apply_pose_blendshape
        self.cano_pose_type = cano_pose_type
        self.dense_sample(body_face_ratio, dense_sample_points)
        self._smplx_init()

        self.input_params = {
            "human_model_path": human_model_path,
            "gender": gender,
            "subdivide_num": subdivide_num,
            "expr_param_dim": expr_param_dim,
            "shape_param_dim": shape_param_dim,
            "cano_pose_type": cano_pose_type,
            "body_face_ratio": body_face_ratio,
            "dense_sample_points": dense_sample_points,
            "apply_pose_blendshape": apply_pose_blendshape,
        }

    def compute_verts_normal(self, verts, faces):
        """
        Compute vertex normals from mesh vertex and face indices.

        Args:
            verts (torch.Tensor): Vertices of the mesh of shape [N, 3].
            faces (np.ndarray or torch.Tensor): Triangular face indices of shape [F, 3].

        Returns:
            torch.Tensor: Per-vertex normals of shape [N, 3], normalized to unit length.
        """

        faces = torch.from_numpy(faces).long().to(verts.device)
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
        v0, v1, v2 = verts[i0], verts[i1], verts[i2]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
        vn = torch.zeros_like(verts)

        for idx in [i0, i1, i2]:
            vn.scatter_add_(0, idx[:, None].repeat(1, 3), face_normals)

        # Handle zero normals
        mask = (vn * vn).sum(-1) > 1e-20
        vn[mask] = F.normalize(vn[mask], dim=-1)
        vn[~mask] = torch.tensor([0.0, 0.0, 1.0], device=vn.device)
        return vn

    def dense_sample(self, body_face_ratio, dense_sample_points):
        """
        Densely samples points from the body and face regions of the canonical mesh,
        according to the specified ratio and target number of dense sample points.

        Args:
            body_face_ratio (int): The ratio of body to face sample points. For every
                (body_face_ratio + 1) points, body receives 'body_face_ratio' and face receives 1.
            dense_sample_points (int): Total number of sample points to generate.

        Note:
            - If cached sample points exist, loads from file.
            - Otherwise, reconstructs and samples body and face mesh regions separately,
              concatenates, builds a mask for body points, and saves to cache for future use.

        Side effects:
            - Sets self.dense_pts: torch.Tensor of sampled points.
            - Sets self.is_body: torch.Tensor of bools (mask) with '1' for body points, '0' for head.
        """
        cache_path = f"./pretrained_models/dense_sample_points/{self.cano_pose_type}_{dense_sample_points}.ply"

        if os.path.exists(cache_path):
            pts, _ = load_ply(cache_path)
            body_pts = int(
                (dense_sample_points // (body_face_ratio + 1)) * body_face_ratio
            )

            self.is_body = torch.zeros(pts.shape[0])
            self.is_body[:body_pts] = 1
            self.dense_pts = pts
        else:
            raise FileNotFoundError(f"Dense sample points file not found: {cache_path}, Please Download Prior Model First")

    def rebuild_mesh(self, v, vertices_id, faces_id, num_dense_samples):
        """
        Rebuilds a mesh for the specified set of vertex and face indices, and densely samples points on its surface.

        Args:
            v (torch.Tensor or np.ndarray): The collection of vertices from which to select.
            vertices_id (array-like): The indices of the vertices to include in the mesh.
            faces_id (array-like): The face indices (triplets) indicating the connectivity of mesh faces.
            num_dense_samples (int): The number of dense sample points to extract from the mesh surface.

        Returns:
            torch.Tensor: Surface-sampled points as a tensor of shape (num_dense_samples, 3).
        """

        choice_vertices = v[vertices_id]

        new_mapping = dict()

        for new_id, vertice_id in enumerate(vertices_id):
            new_mapping[vertice_id] = new_id

        faces_id_list = faces_id.reshape(-1).tolist()

        new_faces_id = []
        for face_id in faces_id_list:
            new_faces_id.append(new_mapping[face_id])
        new_faces_id = torch.from_numpy(np.array(new_faces_id).reshape(faces_id.shape))

        mymesh = Mesh(v=choice_vertices, f=new_faces_id)

        dense_sample_pts = mymesh.sample_surface(num_dense_samples).detach().cpu()

        return dense_sample_pts

    def get_neutral_pose_human(
        self, jaw_zero_pose, use_id_info, shape_param, device, face_offset, joint_offset
    ):
        """
        Returns the neutral (canonical) pose of the human mesh given shape, expression, and optional identity information.

        Args:
            jaw_zero_pose (bool): If True, the jaw pose is set to zero (closed mouth); otherwise uses the default neutral pose.
            use_id_info (bool): Whether to use shape, face, and joint identity information for the mesh.
            shape_param (torch.Tensor): The shape (beta) parameters for the body, with shape (B, shape_param_dim).
            device (torch.device or str): Device on which to allocate tensors and perform computation.
            face_offset (torch.Tensor or None): Face mesh offset (if `use_id_info` is True), else None.
            joint_offset (torch.Tensor or None): Joint offset (if `use_id_info` is True), else None.

        Returns:
            dict: Output dictionary from the SMPL-X layer's forward pass containing mesh vertices, joints, and other model outputs,
                  typically with fields like 'vertices', 'joints', etc., representing the neutral (canonical) body pose.
        """

        smpl_x = self.base_skinning
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        neutral_body_pose = (
            smpl_x.neutral_body_pose.view(1, -1).repeat(batch_size, 1).to(device)
        )  # neutral (canonical) body pose
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        if jaw_zero_pose:
            jaw_pose = torch.zeros((batch_size, 3)).float().to(device)
        else:
            jaw_pose = (
                smpl_x.neutral_jaw_pose.view(1, 3).repeat(batch_size, 1).to(device)
            )  # open mouth

        if use_id_info:
            shape_param = shape_param
            # face_offset = smpl_x.face_offset[None,:,:].float().to(device)
            # joint_offset = smpl_x.get_joint_offset(self.joint_offset[None,:,:])
            face_offset = face_offset
            joint_offset = (
                smpl_x.get_joint_offset(joint_offset)
                if joint_offset is not None
                else None
            )

        else:
            shape_param = (
                torch.zeros((batch_size, smpl_x.shape_param_dim)).float().to(device)
            )
            face_offset = None
            joint_offset = None

        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=neutral_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=jaw_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        vertices = output.vertices[0]
        faces = self.smplx_layer.faces
        # trimesh.Trimesh(vertices[0], faces, process=False).export("fuck.obj")

        verts_normal = self.compute_verts_normal(vertices, faces)
        query_verts_normal = verts_normal[self.query_indx]

        # using dense sample strategy, and warp to neutral pose
        mesh_neutral_pose_upsampled = self.upsample_mesh_batch(
            smpl_x,
            shape_param=shape_param,
            neutral_body_pose=neutral_body_pose,
            jaw_pose=jaw_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
            device=device,
        )

        mesh_neutral_pose = output.vertices
        joint_neutral_pose = output.joints[
            :, : smpl_x.joint_num, :
        ]  # neutral (canonical) human joints [B, 55, 3]

        # compute transformation matrix for mapping from neutral pose to zero pose
        neutral_body_pose = neutral_body_pose.view(
            batch_size, len(smpl_x.joint_part["body"]) - 1, 3
        )
        zero_hand_pose = zero_hand_pose.view(
            batch_size, len(smpl_x.joint_part["lhand"]), 3
        )

        neutral_body_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(neutral_body_pose))
        )
        jaw_pose_inv = matrix_to_axis_angle(
            torch.inverse(axis_angle_to_matrix(jaw_pose))
        )

        zero_pose = zero_pose.unsqueeze(1)
        jaw_pose_inv = jaw_pose_inv.unsqueeze(1)

        pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose_inv,
                jaw_pose_inv,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )

        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]

        # transform_mat_neutral_pose is a function to warp neutral pose to zero pose  (neutral pose is *-pose)
        _, transform_mat_neutral_pose = batch_rigid_transform(
            pose[:, :, :, :], joint_neutral_pose[:, :, :], self.smplx_layer.parents
        )  # [B, 55, 4, 4]

        return (
            mesh_neutral_pose_upsampled,
            mesh_neutral_pose,
            query_verts_normal[None].repeat(batch_size, 1, 1),
            transform_mat_neutral_pose,
        )

    def upsample_mesh_batch(
        self,
        smpl_x,
        shape_param,
        neutral_body_pose,
        jaw_pose,
        expression,
        betas,
        face_offset=None,
        joint_offset=None,
        device=None,
    ):
        """
        Upsample the SMPL-X/FLAME mesh for a batch of shape, pose, and expression parameters.

        Args:
            smpl_x: The SMPL-X/FLAME model instance for mesh and joint computation.
            shape_param (torch.Tensor): Batch of body shape parameters [B, shape_param_dim].
            neutral_body_pose (torch.Tensor): Batch of neutral body poses [B, body_pose_dim=21*3].
            jaw_pose (torch.Tensor): Batch of jaw pose parameters [B, 3].
            expression (torch.Tensor): Batch of facial expression parameters [B, expr_param_dim].
            betas (torch.Tensor): Batch of shape coefficients [B, shape_param_dim].
            face_offset (torch.Tensor, optional): Optional face-specific offsets [B, face_vertex_count, 3].
            joint_offset (torch.Tensor, optional): Optional joint location offsets [B, joint_count, 3].
            device (str or torch.device, optional): Target device ("cpu", "cuda", etc.)

        Returns:
            posed_mesh_verts (torch.Tensor): Upsampled mesh vertices in posed space [B, N, 3].
            posed_mesh_faces (np.ndarray): Mesh face indices after upsampling [F, 3].
            dense_points (torch.Tensor): Upsampled canonical mesh surface points [B, N, 3].
            transform_mat_vertex (torch.Tensor): Per-vertex transformation matrices [B, N, 4, 4].
        """

        device = device if device is not None else avaliable_device()

        batch_size = shape_param.shape[0]
        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )

        dense_pts = self.dense_pts.to(device)
        dense_pts = dense_pts.unsqueeze(0).repeat(expression.shape[0], 1, 1)

        blend_shape_offset = blend_shapes(betas, self.shape_dirs)

        dense_pts = dense_pts + blend_shape_offset

        joint_zero_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        neutral_pose = torch.cat(
            (
                zero_pose,
                neutral_body_pose,
                jaw_pose,
                zero_pose,
                zero_pose,
                zero_hand_pose,
                zero_hand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]

        neutral_pose = axis_angle_to_matrix(
            neutral_pose.view(-1, 55, 3)
        )  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint = batch_rigid_transform(
            neutral_pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )

        skinning_weight = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # B 55 4,4, B N 55 -> B N 4 4
        transform_mat_vertex = torch.einsum(
            "blij,bnl->bnij", transform_mat_joint, skinning_weight
        )

        mesh_neutral_pose_upsampled = self.lbs(dense_pts, transform_mat_vertex, None)

        return mesh_neutral_pose_upsampled

    def lbs(self, xyz, transform_mat_vertex, trans):
        """
        Linear Blend Skinning (LBS) for mesh deformation.

        Args:
            xyz (torch.Tensor): Input points in canonical (neutral) space of shape [B, N, 3].
            transform_mat_vertex (torch.Tensor): Per-vertex LBS transformation matrices of shape [B, N, 4, 4].
            trans (torch.Tensor or None): Optional global translation to be applied, shape [B, 3]. If None, no translation is applied.

        Returns:
            torch.Tensor: Posed mesh points of shape [B, N, 3].
        """

        batch_size = xyz.shape[0]
        num_upsampled = xyz.shape[1]
        xyz = torch.cat(
            (xyz, torch.ones_like(xyz[:, :, :1])), dim=-1
        )  # homogeneous coordinates xyz1 [B, N, 4]
        xyz = torch.matmul(transform_mat_vertex, xyz[:, :, :, None]).view(
            batch_size, num_upsampled, 4
        )[
            :, :, :3
        ]  # [B, N, 3]
        if trans is not None:
            xyz = xyz + trans.unsqueeze(1)
        return xyz

    def get_zero_pose_human(
        self, shape_param, device, face_offset, joint_offset, return_mesh=False
    ):
        """
        Compute the SMPL-X/FLAME neutral (zero) pose joint locations for the given shape (beta) parameters.

        Args:
            shape_param (torch.Tensor): SMPL-X/FLAME shape parameters (betas) of shape [B, shape_param_dim].
            device (torch.device or str): Device to place computation tensors on ("cpu" or "cuda").
            face_offset (torch.Tensor or None): Optional per-batch facial landmark offsets [B, Nf, 3], or None.
            joint_offset (torch.Tensor or None): Optional per-batch joint offsets [B, Nj, 3], or None.
            return_mesh (bool, optional): If True, also return the neutral mesh; otherwise, only return joint positions.

        Returns:
            torch.Tensor: Zero-pose human joint positions of shape [B, joint_num, 3] if return_mesh==False.
            If return_mesh==True, raises NotImplementedError.
        """

        smpl_x = self.base_skinning
        batch_size = shape_param.shape[0]

        zero_pose = torch.zeros((batch_size, 3)).float().to(device)
        zero_body_pose = (
            torch.zeros((batch_size, (len(smpl_x.joint_part["body"]) - 1) * 3))
            .float()
            .to(device)
        )
        zero_hand_pose = (
            torch.zeros((batch_size, len(smpl_x.joint_part["lhand"]) * 3))
            .float()
            .to(device)
        )
        zero_expr = torch.zeros((batch_size, smpl_x.expr_param_dim)).float().to(device)

        face_offset = face_offset
        joint_offset = (
            smpl_x.get_joint_offset(joint_offset) if joint_offset is not None else None
        )
        output = self.smplx_layer(
            global_orient=zero_pose,
            body_pose=zero_body_pose,
            left_hand_pose=zero_hand_pose,
            right_hand_pose=zero_hand_pose,
            jaw_pose=zero_pose,
            leye_pose=zero_pose,
            reye_pose=zero_pose,
            expression=zero_expr,
            betas=shape_param,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )
        joint_zero_pose = output.joints[:, : smpl_x.joint_num, :]  # zero pose human

        if not return_mesh:
            return joint_zero_pose
        else:
            raise NotImplementedError

    def query_voxel_skinning_weights(self, vs):
        """
        Query voxel-based skinning weights for a set of 3D points.

        Args:
            vs (torch.Tensor): Query points of shape [B, N, 3], representing the 3D coordinates in canonical (neutral) space.

        Returns:
            torch.Tensor: Corresponding per-point voxel-based skinning weights of shape [B, N, num_joints].
        """

        bbox = self.voxel_bbox
        scale = bbox[..., 1] - bbox[..., 0]
        center = bbox.mean(dim=1)

        # Normalize vertices to [-1, 1] range
        norm_vs = ((vs - center[None, None]) / scale[None, None]) * 2

        B, N, _ = norm_vs.shape
        weights = (
            F.grid_sample(
                self.voxel_ws.unsqueeze(0),
                norm_vs.view(1, 1, 1, -1, 3).to(self.voxel_ws),
                align_corners=True,
                padding_mode="border",
            )
            .view(B, -1, N)
            .permute(0, 2, 1)
        )

        return weights

    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        """
        Compute per-point transformation matrices using voxel-blended skinning weights.

        Args:
            transform_mat_joint (torch.Tensor): Per-joint transformation matrices of shape [B, num_joints, 4, 4].
            query_points (torch.Tensor): Query points in canonical space of shape [B, N, 3].
            fix_mask (torch.Tensor): Boolean mask of shape [B, N] indicating which points should use the original mesh skinning weights
                                     (e.g., fixed mesh vertices or special semantic parts).

        Returns:
            torch.Tensor: Per-point blended transformation matrices of shape [B, N, 4, 4], where each [4, 4] matrix is a result of
                          blending the per-joint transformations by the voxel-based weights.
        """

        B = transform_mat_joint.shape[0]
        query_weights = self.query_voxel_skinning_weights(query_points.view(1, -1, 3))
        query_weights = query_weights.view(B, -1, query_weights.shape[-1])

        # Use original weights for fixed parts
        orig_weights = self.skinning_weight.unsqueeze(0).repeat(B, 1, 1)
        query_weights[fix_mask] = orig_weights[fix_mask]

        return torch.matmul(
            query_weights, transform_mat_joint.view(B, self.base_skinning.joint_num, 16)
        ).view(B, -1, 4, 4)

    def get_transform_mat_joint(
        self, transform_mat_neutral_pose, joint_zero_pose, smplx_param
    ):
        """
        Args:
            transform_mat_neutral_pose (_type_): [B, 55, 4, 4]
            joint_zero_pose (_type_): [B, 55, 3]
            smplx_param (_type_): dict
        Returns:
            _type_: _description_
        """

        # 1. neutral (canonical) pose -> zero pose
        transform_mat_joint_1 = transform_mat_neutral_pose

        # 2. zero pose -> image pose
        root_pose = smplx_param["root_pose"]
        body_pose = smplx_param["body_pose"]
        jaw_pose = smplx_param["jaw_pose"]
        leye_pose = smplx_param["leye_pose"]
        reye_pose = smplx_param["reye_pose"]
        lhand_pose = smplx_param["lhand_pose"]
        rhand_pose = smplx_param["rhand_pose"]
        # trans = smplx_param['trans']

        # forward kinematics

        pose = torch.cat(
            (
                root_pose.unsqueeze(1),
                body_pose,
                jaw_pose.unsqueeze(1),
                leye_pose.unsqueeze(1),
                reye_pose.unsqueeze(1),
                lhand_pose,
                rhand_pose,
            ),
            dim=1,
        )  # [B, 55, 3]
        pose = axis_angle_to_matrix(pose)  # [B, 55, 3, 3]
        posed_joints, transform_mat_joint_2 = batch_rigid_transform(
            pose[:, :, :, :], joint_zero_pose[:, :, :], self.smplx_layer.parents
        )
        transform_mat_joint_2 = transform_mat_joint_2  # [B, 55, 4, 4]

        # 3. combine: (1) neutral -> zero pose and (2) zero pose -> image pose
        if transform_mat_joint_1 is not None:
            transform_mat_joint = torch.matmul(
                transform_mat_joint_2, transform_mat_joint_1
            )  # [B, 55, 4, 4]
        else:
            transform_mat_joint = transform_mat_joint_2

        return transform_mat_joint, posed_joints

    def get_transform_mat_vertex(self, transform_mat_joint, query_points, fix_mask):
        """
        Computes per-query-point linear blend skinning (LBS) transformation matrices by blending transformation matrices from each joint according to voxel-based skinning weights.

        Args:
            transform_mat_joint (torch.Tensor): Per-joint transformation matrices of shape [B, num_joints, 4, 4].
            query_points (torch.Tensor): Query points coordinates, shape [B, N, 3] or [N, 3].
            fix_mask (torch.Tensor): Boolean mask indicating which query points should use original skinning weights (from canonical mesh), shape [B, N] or [N].

        Returns:
            torch.Tensor: Per-point transformation matrices after skinning weight blending, shape [B, N, 4, 4].
        """

        B = transform_mat_joint.shape[0]
        query_weights = self.skinning_weights(query_points, fix_mask, B)

        return torch.matmul(
            query_weights, transform_mat_joint.view(B, self.base_skinning.joint_num, 16)
        ).view(B, -1, 4, 4)

    def skinning_weights(self, query_points, fix_mask, batch_size):
        """
        Per-point skinning weights over all joints: the voxel field, and the mesh's own weights
        where ``fix_mask`` is set (see ``fixed_part_mask``).

        Returns:
            torch.Tensor: [B, N, num_joints].
        """

        query_weights = self.query_voxel_skinning_weights(query_points.view(1, -1, 3))
        query_weights = query_weights.view(batch_size, -1, query_weights.shape[-1])

        # Use original weights for fixed parts
        orig_weights = self.skinning_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_weights[fix_mask] = orig_weights[fix_mask]
        return query_weights

    def fixed_part_mask(self, batch_size):
        """Hands and face keep the mesh's own skinning weights. Returns [B, N] bool."""

        return (
            ((self.is_rhand + self.is_lhand + self.is_face) > 0)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

    def transform_to_posed_verts_from_neutral_pose(self, points, smplx_data, device):
        """
        Transform the mean 3D vertices to posed vertices from the neutral pose.

            mean_3d (torch.Tensor): Mean 3D vertices with shape [B*Nv, N, 3] + offset.
            smplx_data (dict): SMPL-X data containing body_pose with shape [B*Nv, 21, 3] and betas with shape [B, 100].
            mesh_neutral_pose (torch.Tensor): Mesh vertices in the neutral pose with shape [B*Nv, N, 3].
            transform_mat_neutral_pose (torch.Tensor): Transformation matrix of the neutral pose with shape [B*Nv, 4, 4].
            device (torch.device): Device to perform the computation.

        Returns:
           Points:
        """

        mean_3d = points["neutral_coords"]

        transform_mat = points["transform_mat_to_null_pose"]
        batch_size = mean_3d.shape[0]

        shape_param = self._expand_for_views(smplx_data["betas"], batch_size)
        face_offset = self._expand_for_views(smplx_data.get("face_offset"), batch_size)
        joint_offset = self._expand_for_views(
            smplx_data.get("joint_offset"), batch_size
        )

        if shape_param.shape[0] != batch_size:
            num_views = batch_size // shape_param.shape[0]
            # print(shape_param.shape, batch_size)
            shape_param = (
                shape_param.unsqueeze(1)
                .repeat(1, num_views, 1)
                .view(-1, shape_param.shape[1])
            )
            if face_offset is not None:
                face_offset = (
                    face_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *face_offset.shape[1:])
                )
            if joint_offset is not None:
                joint_offset = (
                    joint_offset.unsqueeze(1)
                    .repeat(1, num_views, 1, 1)
                    .view(-1, *joint_offset.shape[1:])
                )

        # Handle expression offsets
        expr_offset = 0.0
        if "expr" in smplx_data:
            expr_offset = (
                smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
            ).sum(-1)

        try:
            mean_3d = mean_3d + expr_offset
        except:
            import pdb

            pdb.set_trace()

        mask = self.fixed_part_mask(batch_size)

        # Transform to null pose
        transform_mat_null = self.get_transform_mat_vertex(transform_mat, mean_3d, mask)
        null_mean_3d = self.lbs(
            mean_3d, transform_mat_null, torch.zeros_like(smplx_data["trans"])
        )

        # Apply shape blendshapes
        blend_shape = blend_shapes(smplx_data["betas"], self.shape_dirs)
        null_mean_3d += blend_shape

        # Get joint transformations
        joint_null_pose = self.get_zero_pose_human(
            shape_param=shape_param,
            device=device,
            face_offset=face_offset,
            joint_offset=joint_offset,
        )

        transform_mat_joint, _ = self.get_transform_mat_joint(
            None, joint_null_pose, smplx_data
        )
        transform_mat_vertex = self.get_transform_mat_vertex(
            transform_mat_joint, mean_3d, mask
        )

        # Final posed vertices
        posed_mean_3d = self.lbs(
            null_mean_3d, transform_mat_vertex, smplx_data["trans"]
        )

        # Combine transformations
        points["posed_coords"] = posed_mean_3d
        points["transform_mat_posed"] = torch.matmul(
            transform_mat_vertex, transform_mat_null
        )

        return points

    def transform_to_posed_verts(self, smplx_data, device):
        """
        Transforms canonical (neutral-pose) mesh vertices to the posed (deformed/animated) state using SMPL-X parameters.

        Args:
            smplx_data (dict): Dictionary containing SMPL-X parameter tensors, expected keys include:
                - "betas": Shape parameters [B, shape_param_dim]
                - "body_pose": Body pose parameters [B, (body_joint_num-1)*3]
                - "trans": Global translation [B, 3]
                - "expr": Expression parameters [B, expr_param_dim]
                - "face_offset" (optional): Face vertex offset [B, Nf, 3]
                - "joint_offset" (optional): Joint offset [B, Nj, 3]
            device (torch.device or str): Target device for computations.

        Returns:
            mean_3d (torch.Tensor): Posed mesh vertices of shape [B, N, 3].
            transform_matrix (torch.Tensor): Final transformation matrices applied to mesh vertices [B, N, 4, 4].
        """

        raise NotImplementedError
        # neutral posed verts
        # points = self.get_query_points(
        #     smplx_data, device
        # )
        # mean_3d, transform_matrix = self.transform_to_posed_verts_from_neutral_pose(
        #     mesh_neutral_pose,
        #     smplx_data,
        #     mesh_neutral_pose,
        #     transform_mat_neutral_pose,
        #     device,
        # )

        # return mean_3d, transform_matrix

    def transform_to_neutral_pose(
        self, mean_3d, smplx_data, mesh_neutral_pose, transform_mat_neutral_pose, device
    ):
        """
        Transform vertices from posed state to neutral pose.

        Args:
            mean_3d: Mean 3D vertices with shape [B*Nv, N, 3] + offset
            smplx_data: SMPL-X data containing:
                - body_pose: [B*Nv, 21, 3]
                - betas: [B, 100]
                - expr: expression parameters
                - face_offset: optional face offset
                - joint_offset: optional joint offset
            mesh_neutral_pose: Mesh vertices in neutral pose [B*Nv, N, 3]
            transform_mat_neutral_pose: Neutral pose transformation matrix [B*Nv, 4, 4]
            device: Computation device

        Returns:
            Posed vertices with shape [B*Nv, N, 3] + offset
        """
        batch_size = mean_3d.shape[0]

        # Process shape parameters and offsets for multi-view cases
        shape_param = self._expand_for_views(smplx_data["betas"], batch_size)
        face_offset = self._expand_for_views(smplx_data.get("face_offset"), batch_size)
        joint_offset = self._expand_for_views(
            smplx_data.get("joint_offset"), batch_size
        )

        # Apply expression offset
        expr_offset = (
            smplx_data["expr"].unsqueeze(1).unsqueeze(1) * self.expr_dirs
        ).sum(
            -1
        )  # [B, 1, 1, 50] x [N_V, 3, 50] -> [B, N_v, 3]
        mean_3d = mean_3d + expr_offset

        raise NotImplementedError

    def obtain_query_info(self, batch_size=1):
        """
        Collect and return meta information about query points used by the SMPL-X model.

        Args:
            batch_size (int, optional): Batch size for potential future extensions. Defaults to 1.

        Returns:
            dict: Dictionary containing query point meta data, such as skinning weights, pose and expression directions,
                  body part segmentation flags, and special indices/flags for hands, face, body, etc.
        """

        mesh_smplx_meta = dict(
            skinning_weight=self.skinning_weight,
            pose_dirs=self.pose_dirs,
            expr_dirs=self.expr_dirs,
            shape_dirs=self.shape_dirs,
            is_cavity=self.is_cavity,
            query_indx=self.query_indx,
            is_rhand=self.is_rhand,
            is_lhand=self.is_lhand,
            is_face=self.is_face,
            is_face_expr=self.is_face_expr,
            is_lower_body=self.is_lower_body,
            is_upper_body=self.is_upper_body,
            is_constrain_body=self.is_constrain_body,
        )

        return mesh_smplx_meta

    def get_query_points(self, get_query_pts, smplx_data, device):
        """
        Retrieves query points and relevant meta data for a given set of SMPL-X parameters.

        Args:
            get_query_pts: (Unused) Argument placeholder for interface compatibility.
            smplx_data (dict): Dictionary containing SMPL-X parameters, such as 'betas', 'face_offset', and 'joint_offset'.
            device (str or torch.device): Device to perform computations on.

        Returns:
            dict: A dictionary containing:
                - 'neutral_coords' (torch.Tensor): The neutral pose coordinates.
                - 'ori_neutral_coords' (torch.Tensor): The original (unupsampled) neutral pose coordinates.
                - 'neutral_normals' (torch.Tensor): The vertex normals in the neutral pose.
                - 'transform_mat_to_null_pose' (torch.Tensor): Transformation matrices for mapping to the neutral pose.
                - 'mesh_meta' (dict): Query point meta information.
        """

        batch_size = smplx_data["betas"].shape[0]

        results = self.get_neutral_pose_human(
            jaw_zero_pose=True,
            use_id_info=False,  # we blendshape at zero-pose
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )
        mesh_meta = self.obtain_query_info(batch_size)

        return dict(
            neutral_coords=results[0],
            ori_neutral_coords=results[1],
            neutral_normals=results[2],
            transform_mat_to_null_pose=results[3],
            mesh_meta=mesh_meta,
        )

    def get_body_infos(self):
        """Get indices for different body parts."""
        return {
            "head": torch.where(self.is_face)[0],
            "body": torch.where(~self.is_face)[0],
            "lower_body": torch.where(self.is_lower_body)[0],
            "upper_body": torch.where(self.is_upper_body)[0],
            "hands": torch.cat(
                [
                    torch.where(self.is_rhand)[0],
                    torch.where(self.is_lhand)[0],
                ]
            ),
            "constrain_body": torch.where(self.is_constrain_body)[0],
        }

    ############################################ Auxiliary  Function ########################################
    def _smplx_init(self):
        """Initialize upsampled SMPLX model with registered buffers."""
        smpl_x = self.base_skinning
        dense_pts = self.dense_pts.cuda()
        template_verts = self.smplx_layer.v_template
        faces = self.smplx_layer.faces

        # Compute normals and nearest neighbors
        verts_normal = self.compute_verts_normal(template_verts, faces)
        nn_indices = knn_points(
            dense_pts.unsqueeze(0).cuda(),
            template_verts.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        ).idx.squeeze(0, -1)

        # Helper function to query upsampled values
        def _query(data, indices=nn_indices):
            return data.squeeze(0)[indices.detach().cpu()]

        # Query various properties
        skinning_weight = _query(self.smplx_layer.lbs_weights.float())

        # Prepare pose, expression and shape directions
        pose_dirs = self.smplx_layer.posedirs.permute(1, 0).view(
            -1, (smpl_x.joint_num - 1) * 9
        )
        expr_dirs = self.smplx_layer.expr_dirs.view(-1, 3 * smpl_x.expr_param_dim)
        shape_dirs = self.smplx_layer.shapedirs.view(-1, 3 * smpl_x.shape_param_dim)

        # Create body part masks
        part_masks = {
            "is_rhand": smpl_x.rhand_vertex_idx,
            "is_lhand": smpl_x.lhand_vertex_idx,
            "is_face": smpl_x.face_vertex_idx,
            "is_face_expr": smpl_x.expr_vertex_idx,
            "is_lower_body": smpl_x.lower_body_vertex_idx,
            "is_upper_body": smpl_x.upper_body_vertex_idx,
            "is_constrain_body": smpl_x.constrain_body_vertex_idx,
        }

        masks = {name: torch.zeros(smpl_x.vertex_num, 1).float() for name in part_masks}
        for name, idx in part_masks.items():
            masks[name][idx] = 1.0

        # Query and register all buffers
        self._register_buffers(
            skinning_weight=skinning_weight,
            pose_dirs=_query(pose_dirs).view(-1, 3, (smpl_x.joint_num - 1) * 3),
            expr_dirs=_query(expr_dirs).view(-1, 3, smpl_x.expr_param_dim),
            shape_dirs=_query(shape_dirs).view(-1, 3, smpl_x.shape_param_dim),
            v_normal=_query(verts_normal),
            is_cavity=_query(torch.FloatTensor(smpl_x.is_cavity)[:, None])[:, 0] > 0,
            query_indx=nn_indices.detach().cpu(),
            **{k: _query(v)[:, 0] > 0 for k, v in masks.items()},
        )

        self.vertex_num_upsampled = dense_pts.shape[0]
        self.base_skinning.vertex_num_upsampled = self.vertex_num_upsampled

        # Initialize voxel skinning
        voxel_ws, voxel_bbox = self._voxel_skinning_init(voxel_size=192)
        if voxel_ws is not None:
            self.register_buffer("voxel_ws", voxel_ws)
            self.register_buffer("voxel_bbox", voxel_bbox)

    def _register_buffers(self, **buffers):
        """Helper to register multiple buffers."""
        for name, data in buffers.items():
            self.register_buffer(name, data.contiguous())

    def _voxel_skinning_init(self, scale_ratio=1.05, voxel_size=256):

        skinning_weight = self.smplx_layer.lbs_weights.float()

        smplx_data = {"betas": torch.zeros(1, self.base_skinning.shape_param_dim)}
        device = skinning_weight.device

        _, mesh_neutral_pose_wo_upsample, _, _ = self.get_neutral_pose_human(
            jaw_zero_pose=True,
            use_id_info=True,
            shape_param=smplx_data["betas"],
            device=device,
            face_offset=smplx_data.get("face_offset", None),
            joint_offset=smplx_data.get("joint_offset", None),
        )

        template_verts = mesh_neutral_pose_wo_upsample.squeeze(0)

        def scale_voxel_size(template_verts, scale_ratio=1.0):
            min_values, _ = torch.min(template_verts, dim=0)
            max_values, _ = torch.max(template_verts, dim=0)

            center = (min_values + max_values) / 2
            size = max_values - min_values

            scale_size = size * scale_ratio

            upper = center + scale_size / 2
            bottom = center - scale_size / 2

            return torch.cat([bottom[:, None], upper[:, None]], dim=1)

        mini_size_bbox = scale_voxel_size(template_verts, scale_ratio)
        z_voxel_size = voxel_size // 2

        # build coordinate
        x_range = np.linspace(0, voxel_size - 1, voxel_size) / (
            voxel_size - 1
        )  # from 0 to 1 with voxel_size uniformly sampled points
        y_range = np.linspace(0, voxel_size - 1, voxel_size) / (voxel_size - 1)
        z_range = np.linspace(0, z_voxel_size - 1, z_voxel_size) / (z_voxel_size - 1)

        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        coordinates = torch.from_numpy(np.stack([x, y, z], axis=-1))

        coordinates[..., 0] = mini_size_bbox[0, 0] + coordinates[..., 0] * (
            mini_size_bbox[0, 1] - mini_size_bbox[0, 0]
        )
        coordinates[..., 1] = mini_size_bbox[1, 0] + coordinates[..., 1] * (
            mini_size_bbox[1, 1] - mini_size_bbox[1, 0]
        )
        coordinates[..., 2] = mini_size_bbox[2, 0] + coordinates[..., 2] * (
            mini_size_bbox[2, 1] - mini_size_bbox[2, 0]
        )

        coordinates = coordinates.view(-1, 3).float()
        coordinates = coordinates.cuda()

        if os.path.exists(f"./pretrained_models/voxel_grid/voxel_{voxel_size}.pth"):
            print(f"load voxel_grid voxel_{voxel_size}.pth")
            voxel_flat = torch.load(
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
                map_location=avaliable_device(),
            )
        else:
            voxel_flat = self._voxel_smooth_register(
                coordinates, template_verts, skinning_weight, k=1, smooth_n=3000
            )

            torch.save(
                voxel_flat,
                os.path.join(f"pretrained_models/voxel_grid/voxel_{voxel_size}.pth"),
            )

        N, LBS_F = voxel_flat.shape

        # x, y, z, C
        voxel_grid_original = voxel_flat.view(
            voxel_size, voxel_size, z_voxel_size, LBS_F
        )

        voxel_grid = voxel_grid_original.permute(3, 2, 1, 0)

        return voxel_grid, mini_size_bbox

    def _expand_for_views(self, tensor, target_batch_size):
        """Expand tensor to match target batch size for multi-view cases."""
        if tensor is None:
            return None

        if tensor.shape[0] == target_batch_size:
            return tensor

        num_views = target_batch_size // tensor.shape[0]
        return (
            tensor.unsqueeze(1)
            .repeat(1, num_views, *([1] * (tensor.dim() - 1)))
            .view(target_batch_size, *tensor.shape[1:])
        )

    @torch.no_grad()
    def _voxel_smooth_register(
        self, voxel_v, template_v, lbs_weights, k=3, smooth_k=30, smooth_n=3000
    ):
        """Smooth KNN to handle skirt deformation."""

        lbs_weights = lbs_weights.cuda()

        dist = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            template_v.unsqueeze(0).cuda(),
            K=1,
            return_nn=True,
        )
        mesh_dis = torch.sqrt(dist.dists)
        mesh_indices = dist.idx.squeeze(0, -1)
        knn_lbs_weights = lbs_weights[mesh_indices]

        mesh_dis = mesh_dis.squeeze()

        print(f"Using k = {smooth_k}, N={smooth_n} for LBS smoothing")
        # Smooth Skinning

        knn_dis = knn_points(
            voxel_v.unsqueeze(0).cuda(),
            voxel_v.unsqueeze(0).cuda(),
            K=smooth_k + 1,
            return_nn=True,
        )
        voxel_dis = torch.sqrt(knn_dis.dists)
        voxel_indices = knn_dis.idx
        voxel_indices = voxel_indices.squeeze()[:, 1:]
        voxel_dis = voxel_dis.squeeze()[:, 1:]

        knn_weights = 1.0 / (mesh_dis[voxel_indices] * voxel_dis)
        knn_weights = knn_weights / knn_weights.sum(-1, keepdim=True)  # [N, K]

        def dists_to_weights(
            dists: torch.Tensor, low: float = None, high: float = None
        ):
            if low is None:
                low = high
            if high is None:
                high = low
            assert high >= low
            weights = dists.clone()
            weights[dists <= low] = 0.0
            weights[dists >= high] = 1.0
            indices = (dists > low) & (dists < high)
            weights[indices] = (dists[indices] - low) / (high - low)
            return weights

        update_weights = dists_to_weights(mesh_dis, low=0.01).unsqueeze(-1)  # [N, 1]

        from tqdm import tqdm

        for _ in tqdm(range(smooth_n)):
            N, _ = update_weights.shape
            new_lbs_weights_chunk_list = []
            for chunk_i in range(0, N, 1000000):

                knn_weights_chunk = knn_weights[chunk_i : chunk_i + 1000000]
                voxel_indices_chunk = voxel_indices[chunk_i : chunk_i + 1000000]

                new_lbs_weights_chunk = torch.einsum(
                    "nk,nkj->nj",
                    knn_weights_chunk,
                    knn_lbs_weights[voxel_indices_chunk],
                )
                new_lbs_weights_chunk_list.append(new_lbs_weights_chunk)
            new_lbs_weights = torch.cat(new_lbs_weights_chunk_list, dim=0)
            if update_weights is None:
                knn_lbs_weights = new_lbs_weights
            else:
                knn_lbs_weights = (
                    1.0 - update_weights
                ) * knn_lbs_weights + update_weights * new_lbs_weights

        return knn_lbs_weights

    ############################################ Auxiliary  Function ########################################


def test():
    def generate_smplx_point():

        def get_smplx_params(data):
            smplx_params = {}
            smplx_keys = [
                "root_pose",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "lhand_pose",
                "rhand_pose",
                "expr",
                "trans",
                "betas",
            ]
            for k, v in data.items():
                if k in smplx_keys:
                    # print(k, v.shape)
                    smplx_params[k] = data[k].unsqueeze(0).cuda()
            return smplx_params

        def sample_one(data):
            smplx_keys = [
                "root_pose",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "lhand_pose",
                "rhand_pose",
                "trans",
            ]
            for k, v in data.items():
                if k in smplx_keys:
                    # print(k, v.shape)
                    data[k] = data[k][:, 0]
            return data

        human_model_path = "./pretrained_models/human_model_files"
        gender = "neutral"
        subdivide_num = 1
        smplx_model = SMPLXVoxelSkinning(
            human_model_path,
            gender,
            shape_param_dim=10,
            expr_param_dim=100,
            subdivide_num=subdivide_num,
            dense_sample_points=160000,
            cano_pose_type=1,
        )
        smplx_model.to("cuda")

        # save_file = f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.npy"
        save_file = f"debug/smplx_points/smplx_subdivide{subdivide_num}.npy"
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        smplx_data = {}
        smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")
        # mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
        #     smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")
        # )

        # neutral_coords = mesh_neutral_pose,
        # ori_neutral_coords = mesh_neutral_pose_wo_upsample,
        # neutral_normals = mesh_neutral_pose_normal,
        # transform_mat_to_neutral_pose = transform_mat_neutral_pose,

        points = smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")

        debug_pose = torch.load("./debug/pose_example.pth")
        debug_pose["expr"] = torch.FloatTensor([0.0] * 100)

        smplx_data = get_smplx_params(debug_pose)
        smplx_data = sample_one(smplx_data)
        smplx_data["betas"] = torch.ones_like(smplx_data["betas"])

        points = smplx_model.transform_to_posed_verts_from_neutral_pose(
            points,
            smplx_data,
            "cuda",
        )

        posed_coords = points["posed_coords"]

        # get_body_infos = smplx_model.get_body_infos()

        # for key, item in get_body_infos.items():
        #     part_points = posed_coords[0][item]
        #     save_ply(f'posed_verts_{key}.ply', part_points)

        save_ply("posed_verts.ply", posed_coords[0])

        return points

    generate_smplx_point()


def get_query_points():
    def get_smplx_params(data):
        smplx_params = {}
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
            "betas",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                smplx_params[k] = data[k].unsqueeze(0).cuda()
        return smplx_params

    def sample_one(data):
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                data[k] = data[k][:, 0]
        return data

    human_model_path = "./pretrained_models/human_model_files"
    gender = "neutral"
    subdivide_num = 1
    smplx_model = SMPLXVoxelSkinning(
        human_model_path,
        gender,
        shape_param_dim=10,
        expr_param_dim=100,
        subdivide_num=subdivide_num,
        dense_sample_points=160000,
        cano_pose_type=1,
    )
    smplx_model.to("cuda")

    # save_file = f"pretrained_models/human_model_files/smplx_points/smplx_subdivide{subdivide_num}.npy"
    save_file = f"debug/smplx_points/smplx_subdivide{subdivide_num}.npy"
    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    smplx_data = {}
    smplx_data["betas"] = torch.zeros((1, 10)).to(device="cuda")
    # mesh_neutral_pose, mesh_neutral_pose_wo_upsample, transform_mat_neutral_pose = (
    #     smplx_model.get_query_points(smplx_data=smplx_data, device="cuda")
    # )

    # neutral_coords = mesh_neutral_pose,
    # ori_neutral_coords = mesh_neutral_pose_wo_upsample,
    # neutral_normals = mesh_neutral_pose_normal,
    # transform_mat_to_neutral_pose = transform_mat_neutral_pose,

    points = smplx_model.get_query_points(None, smplx_data=smplx_data, device="cuda")

    return points


if __name__ == "__main__":
    test()
