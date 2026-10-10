# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:25:39
# @Function      : Gaussian Splatting Model Class
import math

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import Tensor

from core.models.rendering.utils.sh_utils import RGB2SH, SH2RGB


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


class GaussianModel:
    """
    A class representing a differentiable model of 3D Gaussians for neural rendering.

    The GaussianModel encapsulates per-point attributes such as 3D position (xyz),
    opacity, rotation, scaling, and spherical harmonics coefficients or RGB appearance features.
    It provides functionality for handling attribute activations, device management,
    and efficient cloning of model instances. The model is compatible with torch operations and
    serves as a core structure for Gaussian Splatting-based rendering and optimization pipelines.
    """

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        # rgb activation function
        self.rgb_activation = torch.sigmoid

    def __init__(self, xyz, opacity, rotation, scaling, shs, use_rgb=False) -> None:
        """
        Initialize the GaussianModel instance.

        Args:
            xyz (Tensor): Tensor of shape [N, 3] representing the 3D positions of the Gaussians.
            opacity (Tensor): Tensor of shape [N, C_opacity] representing the opacity values for each Gaussian.
            rotation (Tensor): Tensor of shape [N, C_rotation] representing the rotations for each Gaussian.
            scaling (Tensor): Tensor of shape [N, C_scaling] representing the scale factors for each Gaussian.
            shs (Tensor): Tensor of shape [N, SH_Coeff, 3] containing the spherical harmonics coefficients or RGB features.
            use_rgb (bool, optional): Whether to treat the appearance features as RGB. Default is False.
        """

        self.setup_functions()

        self.xyz: Tensor = xyz
        self.opacity: Tensor = opacity
        self.rotation: Tensor = rotation
        self.scaling: Tensor = scaling
        self.shs: Tensor = shs  # [B, SH_Coeff, 3]

        self.use_rgb = use_rgb  # shs indicates rgb?

    def construct_list_of_attributes(self):
        """
        This function constructs and returns a list of attribute names representing the
        features of each 3D Gaussian in the model. The attribute list includes coordinates,
        normal vector components, spherical harmonics feature names for DC/rest,
        opacity, scaling, and rotation parameter names.
        """

        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]

        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path, extra_vertex_fields=(), extra_elements=()):
        """
        Save the 3D Gaussian data and attributes to a PLY file.

        Args:
            path (str): The file path where the PLY file will be saved.
            extra_vertex_fields: ``(name, numpy dtype, array[N])`` appended after the standard
                vertex properties, e.g. skinning joints and weights.
            extra_elements: ``PlyElement`` written after ``vertex``, e.g. a skeleton.

        This method exports the 3D positions, normals, appearance features (either SH or RGB),
        opacity, scaling, and rotation attributes of all Gaussians in the model into a structured
        PLY format. The exported PLY will include all per-Gaussian attributes and can be used
        for visualization or further processing in 3D graphics tools.
        """

        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)

        if self.use_rgb:
            shs = RGB2SH(self.shs)
        else:
            shs = self.shs

        features_dc = shs[:, :1]
        features_rest = shs[:, 1:]

        f_dc = (
            features_dc.float().detach().flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            features_rest.float()
            .detach()
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = (
            inverse_sigmoid(torch.clamp(self.opacity, 1e-3, 1 - 1e-3))
            .detach()
            .cpu()
            .numpy()
        )

        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        dtype_full += [(name, dtype) for name, dtype, _ in extra_vertex_fields]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        standard = [name for name, _ in dtype_full[: attributes.shape[1]]]
        for i, name in enumerate(standard):
            elements[name] = attributes[:, i]
        for name, _, values in extra_vertex_fields:
            elements[name] = values
        el = PlyElement.describe(elements, "vertex")
        PlyData([el, *extra_elements]).write(path)

    def load_ply(self, path):
        """
        Load the 3D Gaussian data and attributes from a PLY file.

        Args:
            path (str): The file path from which to load the PLY data.

        This method reads the PLY file specified by 'path' and loads the 3D positions,
        appearance features (SH or RGB), opacity, scaling, and rotation
        attributes into the model. The method automatically infers the
        spherical harmonics degree and reconstructs all tensors to be stored
        in the current GaussianModel instance, updating its parameters accordingly.
        """

        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]

        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        sh_degree = int(math.sqrt((len(extra_f_names) + 3) / 3)) - 1

        print("load sh degree: ", sh_degree)

        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        # 0, 3, 8, 15
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz = torch.from_numpy(xyz).to(self.xyz)
        opacities = torch.from_numpy(opacities).to(self.opacity)
        rotation = torch.from_numpy(rots).to(self.rotation)
        scales = torch.from_numpy(scales).to(self.scaling)
        features_dc = torch.from_numpy(features_dc).to(self.shs)
        features_rest = torch.from_numpy(features_extra).to(self.shs)

        shs = torch.cat([features_dc, features_rest], dim=2)

        if self.use_rgb:
            shs = SH2RGB(shs)
        else:
            shs = shs

        self.xyz: Tensor = xyz
        self.opacity: Tensor = self.opacity_activation(opacities)
        self.rotation: Tensor = self.rotation_activation(rotation)
        self.scaling: Tensor = self.scaling_activation(scales)
        self.shs: Tensor = shs.permute(0, 2, 1)

        self.active_sh_degree = sh_degree

    def clone(self):
        """
        Create a deep copy of the GaussianModel instance.

        Returns:
            GaussianModel: A new instance with identical parameter values.
        """
        xyz = self.xyz.clone()
        opacity = self.opacity.clone()
        rotation = self.rotation.clone()
        scaling = self.scaling.clone()
        shs = self.shs.clone()
        use_rgb = self.use_rgb

        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)

    def CloneMaskGaussian(self, gs_deform_scale=0.005):
        """
        Create a deep copy of the GaussianModel instance with only the xyz coordinates.

        Args:
            gs_deform_scale (float, optional): Deformation scale for Gaussian Splatting. Default 0.005.

        Returns:
            GaussianModel: A new instance with only the xyz coordinates.
        """
        """only containng xyz Gaussian!
        """

        xyz = self.xyz.clone()
        # Default settings.
        # opacity = torch.ones_like(self.opacity)
        opacity = self.opacity.clone().detach()
        # Identity mapping
        rotation = torch.ones_like(self.rotation.clone())
        rotation[:, 1:] = 0
        scaling = (
            torch.ones_like(self.scaling) * gs_deform_scale
        )  # This is an empirically determined value
        shs = torch.ones_like(self.shs)
        use_rgb = self.use_rgb

        return GaussianModel(xyz, opacity, rotation, scaling, shs, use_rgb)
