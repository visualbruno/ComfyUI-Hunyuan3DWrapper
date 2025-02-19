# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import tempfile
import os
from typing import Union

import pymeshlab
import trimesh

from .models.vae import Latent2MeshOutput

import folder_paths


def load_mesh(path):
    if path.endswith(".glb"):
        mesh = trimesh.load(path)
    else:
        mesh = pymeshlab.MeshSet()
        mesh.load_new_mesh(path)
    return mesh


def reduce_face(mesh: pymeshlab.MeshSet, max_facenum: int = 200000):
    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return mesh


def remove_floater(mesh: pymeshlab.MeshSet):
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def pymeshlab2trimesh(mesh: pymeshlab.MeshSet):
    # Create temp directory with explicit permissions
    temp_dir = folder_paths.temp_directory
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        temp_path = os.path.join(temp_dir, 'temp_mesh.ply')
        
        # Save and load mesh
        mesh.save_current_mesh(temp_path)
        loaded_mesh = trimesh.load(temp_path)
        
        # Check loaded object type
        if isinstance(loaded_mesh, trimesh.Scene):
            combined_mesh = trimesh.Trimesh()
            # If Scene, iterate through all geometries and combine
            for geom in loaded_mesh.geometry.values():
                combined_mesh = trimesh.util.concatenate([combined_mesh, geom])
            loaded_mesh = combined_mesh
            
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return loaded_mesh
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Error in pymeshlab2trimesh: {str(e)}")


def trimesh2pymeshlab(mesh: trimesh.Trimesh):
    # Create temp directory with explicit permissions
    temp_dir = folder_paths.temp_directory
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        temp_path = os.path.join(temp_dir, 'temp_mesh.ply')
        
        # Handle scene with multiple geometries
        if isinstance(mesh, trimesh.scene.Scene):
            temp_mesh = None
            for idx, obj in enumerate(mesh.geometry.values()):
                if idx == 0:
                    temp_mesh = obj
                else:
                    temp_mesh = temp_mesh + obj
            mesh = temp_mesh
            
        # Export and load mesh
        mesh.export(temp_path)
        mesh_set = pymeshlab.MeshSet()
        mesh_set.load_new_mesh(temp_path)
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return mesh_set
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Error in trimesh2pymeshlab: {str(e)}")


def export_mesh(input, output):
    if isinstance(input, pymeshlab.MeshSet):
        mesh = output
    elif isinstance(input, Latent2MeshOutput):
        output = Latent2MeshOutput()
        output.mesh_v = output.current_mesh().vertex_matrix()
        output.mesh_f = output.current_mesh().face_matrix()
        mesh = output
    else:
        mesh = pymeshlab2trimesh(output)
    return mesh


def import_mesh(mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str]) -> pymeshlab.MeshSet:
    if isinstance(mesh, str):
        mesh = load_mesh(mesh)
    elif isinstance(mesh, Latent2MeshOutput):
        mesh = pymeshlab.MeshSet()
        mesh_pymeshlab = pymeshlab.Mesh(vertex_matrix=mesh.mesh_v, face_matrix=mesh.mesh_f)
        mesh.add_mesh(mesh_pymeshlab, "converted_mesh")

    if isinstance(mesh, (trimesh.Trimesh, trimesh.scene.Scene)):
        mesh = trimesh2pymeshlab(mesh)

    return mesh

def bpt_remesh(self, mesh: trimesh.Trimesh, verbose: bool = False, max_seq_len:int=10000, cond_dim:int=768, pc_num: int=8192, with_normal: bool = True, kwarg_k: int = 50, kwarg_p: float = 0.95):
        from .bpt.model import data_utils
        from .bpt.model.model import MeshTransformer
        from .bpt.model.serializaiton import BPT_deserialize
        from .bpt.utils import sample_pc, joint_filter

        pc_normal = sample_pc(mesh, pc_num=pc_num, with_normal=with_normal)

        pc_normal = pc_normal[None, :, :] if len(pc_normal.shape) == 2 else pc_normal

        from torch.serialization import add_safe_globals
        from deepspeed.runtime.fp16.loss_scaler import LossScaler
        from deepspeed.runtime.zero.config import ZeroStageEnum
        from deepspeed.utils.tensor_fragment import fragment_address

        add_safe_globals([LossScaler, fragment_address, ZeroStageEnum])

        model = MeshTransformer(cond_dim=cond_dim, max_seq_len=max_seq_len)

        comfyui_dir = os.path.dirname(os.path.abspath(__file__)) 
        model_path = os.path.join(comfyui_dir, 'bpt/bpt-8-16-500m.pt')
        print(model_path)
        model.load(model_path)
        model = model.eval()
        model = model.half()
        model = model.cuda()

        import torch
        pc_tensor = torch.from_numpy(pc_normal).cuda().half()
        if len(pc_tensor.shape) == 2:
            pc_tensor = pc_tensor.unsqueeze(0)

        codes = model.generate(
            pc=pc_tensor,
            filter_logits_fn=joint_filter,
            filter_kwargs=dict(k=50, p=0.95),
            return_codes=True,
        )

        coords = []
        try:
            for i in range(len(codes)):
                code = codes[i]
                code = code[code != model.pad_id].cpu().numpy()
                vertices = BPT_deserialize(
                    code,
                    block_size=model.block_size,
                    offset_size=model.offset_size,
                    use_special_block=model.use_special_block,
                )
                coords.append(vertices)
        except:
            coords.append(np.zeros(3, 3))

        # convert coordinates to mesh
        vertices = coords[0]
        faces = torch.arange(1, len(vertices) + 1).view(-1, 3)

        # Move to CPU
        faces = faces.cpu().numpy()

        del model

        return data_utils.to_mesh(vertices, faces, transpose=False, post_process=True)

class BptMesh:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_seq_len: int = 10000,
        cond_dim: int = 768,
        kwarg_k: int = 50,
        kwarg_p: float = 0.95,
        verbose: bool = False 
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        mesh = bpt_remesh(self, mesh=mesh, cond_dim=cond_dim, max_seq_len=max_seq_len, kwarg_k=kwarg_k, kwarg_p=kwarg_p)
        return mesh

class FaceReducer:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
        max_facenum: int = 40000
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh]:
        ms = import_mesh(mesh)
        ms = reduce_face(ms, max_facenum=max_facenum)
        mesh = export_mesh(mesh, ms)
        return mesh


class FloaterRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)
        ms = remove_floater(ms)
        mesh = export_mesh(mesh, ms)
        return mesh


class DegenerateFaceRemover:
    def __call__(
        self,
        mesh: Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput, str],
    ) -> Union[pymeshlab.MeshSet, trimesh.Trimesh, Latent2MeshOutput]:
        ms = import_mesh(mesh)

        # Create temp file with explicit closing
        temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
        temp_file_path = temp_file.name
        temp_file.close()

        try:
            ms.save_current_mesh(temp_file_path)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(temp_file_path)
        finally:
            # Ensure temp file is removed
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass

        mesh = export_mesh(mesh, ms)
        return mesh
