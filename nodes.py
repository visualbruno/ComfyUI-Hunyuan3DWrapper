import os
import torch
from PIL import Image
from pathlib import Path
import numpy as np
import trimesh

from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar

script_directory = os.path.dirname(os.path.abspath(__file__))

from .utils import log, print_memory, pil_list_to_torch_batch

class ComfyProgressCallback:
    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        
    def __call__(self, pipe, i, t, callback_kwargs):
        self.pbar.update(1)
        return {
            "latents": callback_kwargs["latents"],
            "prompt_embeds": callback_kwargs["prompt_embeds"],
            "negative_prompt_embeds": callback_kwargs["negative_prompt_embeds"]
        }

class Hy3DTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_vae": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
            },
        }
    RETURN_TYPES = ("HY3DCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer, compile_vae):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_transformer": compile_transformer,
            "compile_vae": compile_vae,
        }

        return (compile_args, )
    
#region Model loading
class Hy3DModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            },
            "optional": {
                "compile_args": ("HY3DCOMPILEARGS", {"tooltip": "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"}),
            }
        }

    RETURN_TYPES = ("HY3DMODEL",)
    RETURN_NAMES = ("pipeline", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model, compile_args=None):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        config_path = os.path.join(script_directory, "configs", "dit_config.yaml")
        model_path = folder_paths.get_full_path("diffusion_models", model)
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=model_path, 
            config_path=config_path, 
            use_safetensors=True, 
            device=device, 
            offload_device=offload_device,
            compile_args=compile_args)
        return (pipe,)

class DownloadAndLoadHy3DDelightModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-delight-v2-0"],),
            },
        }

    RETURN_TYPES = ("DELIGHTMODEL",)
    RETURN_NAMES = ("delight_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=["*hunyuan3d-delight-v2-0*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

        delight_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        delight_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(delight_pipe.scheduler.config)
        delight_pipe = delight_pipe.to(device, torch.float16)
        delight_pipe.enable_model_cpu_offload()
        
        return (delight_pipe,)

class LoadCustomMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load. Tested only for now with other hunyuan3d-2 glbs"}), 
            }
        }
    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "main"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def main(self, glb):
        
        mesh = trimesh.load(glb, force="mesh")
        
        return (mesh,)
        
class Hy3DDelightImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delight_pipe": ("DELIGHTMODEL",),
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 50, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "cfg_image": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "cfg_text": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, delight_pipe, image, width, height, cfg_image, cfg_text, steps, seed):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)

        image = delight_pipe(
            prompt="",
            image=image,
            generator=torch.manual_seed(seed),
            height=height,
            width=width,
            num_inference_steps=steps,
            image_guidance_scale=cfg_image,
            guidance_scale=cfg_text,
            output_type="pt",
            
        ).images[0]

        out_tensor = image.unsqueeze(0).permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensor, )
    
class DownloadAndLoadHy3DPaintModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-paint-v2-0"],),
            },
        }

    RETURN_TYPES = ("HY3DPAINTMODEL",)
    RETURN_NAMES = ("multiview_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=[f"*{model}*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
        custom_pipeline_path = os.path.join(script_directory, 'hy3dgen', 'texgen', 'hunyuanpaint')

        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline=custom_pipeline_path, 
            torch_dtype=torch.float16)

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
        pipeline.enable_model_cpu_offload()
        return (pipeline,)

class Hy3DCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
            },
        }

    RETURN_TYPES = ("HY3DCAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list
            }
        
        return (camera_config,)
    
class Hy3DMeshUVWrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
            },
        }

    RETURN_TYPES = ("HY3DMESH", )
    RETURN_NAMES = ("mesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh):
        from .hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
        mesh = mesh_uv_wrap(mesh)
        
        return (mesh,)

class Hy3DRenderMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "texture_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MESHRENDER")
    RETURN_NAMES = ("normal_maps", "position_maps", "renderer")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, render_size, texture_size, camera_config=None):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size)

        self.render.load_mesh(mesh)

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        normal_tensors = torch.stack(normal_maps, dim=0)
        
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        position_tensors = torch.stack(position_maps, dim=0)
        
        return (normal_tensors, position_tensors, self.render,)
    
    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='th')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='th')
            position_maps.append(position_map)

        return position_maps
    
class Hy3DSampleMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DPAINTMODEL",),
                "ref_image": ("IMAGE", ),
                "normal_maps": ("IMAGE", ),
                "position_maps": ("IMAGE", ),
                "view_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, ref_image, normal_maps, position_maps, view_size, seed, steps, camera_config=None):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        torch.manual_seed(seed)
        generator=torch.Generator(device=pipeline.device).manual_seed(seed)

        input_image = ref_image.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        device = mm.get_torch_device()

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
        
        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        
        normal_maps_np = (normal_maps * 255).to(torch.uint8).cpu().numpy()
        normal_maps_pil = [Image.fromarray(normal_map) for normal_map in normal_maps_np]

        position_maps_np = (position_maps * 255).to(torch.uint8).cpu().numpy()
        position_maps_pil = [Image.fromarray(position_map) for position_map in position_maps_np]
        
        control_images = normal_maps_pil + position_maps_pil

        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((view_size, view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        callback = ComfyProgressCallback(total_steps=steps)

        multiview_images = pipeline(
            input_image,
            width=view_size,
            height=view_size,
            generator=generator,
            num_in_batch = num_view,
            camera_info_gen = [camera_info],
            camera_info_ref = [[0]],
            normal_imgs = normal_image,
            position_imgs = position_image,
            num_inference_steps=steps,
            output_type="pt",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"]
            ).images

        out_tensors = multiview_images.permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensors,)
    
class Hy3DBakeFromMultiview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER") 
    RETURN_NAMES = ("texture", "mask", "renderer")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, images, renderer, camera_config=None):
        device = mm.get_torch_device()
        self.render = renderer

        multiviews = images.permute(0, 3, 1, 2)
        multiviews = multiviews.cpu().numpy()
        multiviews_pil = [Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8)) for image in multiviews]

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            selected_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            selected_view_weights = camera_config["selected_view_weights"]

        merge_method = 'fast'
        self.bake_exp = 4
        
        texture, mask = self.bake_from_multiview(multiviews_pil,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=merge_method)
        
        
        mask = mask.squeeze(-1).cpu().float()
        texture = texture.unsqueeze(0).cpu().float()

        return (texture, mask, self.render)
    
    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        pbar = ProgressBar(len(views))
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            pbar.update(1)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8
    
class Hy3DMeshVerticeInpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER" ) 
    RETURN_NAMES = ("texture", "mask", "renderer" )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, texture, renderer, mask):
        from .hy3dgen.texgen.differentiable_renderer.mesh_processor import meshVerticeInpaint
        vtx_pos, pos_idx, vtx_uv, uv_idx = renderer.get_mesh()

        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = texture.squeeze(0).cpu().numpy() * 255

        texture_np, mask_np = meshVerticeInpaint(
            texture_np, mask_np, vtx_pos, vtx_uv, pos_idx, uv_idx)
            
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return (texture_tensor, mask_tensor, renderer)

class CV2InpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "inpaint_radius": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "inpaint_method": (["ns", "telea"], {"default": "ns"}),
            },
        }

    RETURN_TYPES = ("IMAGE", ) 
    RETURN_NAMES = ("texture", )
    FUNCTION = "inpaint"
    CATEGORY = "Hunyuan3DWrapper"

    def inpaint(self, texture, mask, inpaint_radius, inpaint_method):
        import cv2
        mask = 1 - mask
        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = (texture.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        if inpaint_method == "ns":
            inpaint_algo = cv2.INPAINT_NS
        elif inpaint_method == "telea":
            inpaint_algo = cv2.INPAINT_TELEA
            
        texture_np = cv2.inpaint(
            texture_np,
            mask_np,
            inpaint_radius,
            inpaint_algo)
        
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)
        
        return (texture_tensor, )
    
class Hy3DApplyTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("HY3DMESH", ) 
    RETURN_NAMES = ("mesh", )
    FUNCTION = "apply"
    CATEGORY = "Hunyuan3DWrapper"

    def apply(self, texture, renderer):
        self.render = renderer
        self.render.set_texture(texture.squeeze(0))
        textured_mesh = self.render.save_mesh()
        
        return (textured_mesh,)
    
class Hy3DGenerateMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DMODEL",),
                "image": ("IMAGE", ),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 16}),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, image, steps, guidance_scale, octree_resolution, seed, mask=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)
        image = image * 2 - 1

        if mask is not None:
            mask = mask.unsqueeze(0).to(device)

        pipeline.to(device)

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        mesh = pipeline(
            image=image, 
            mask=mask,
            num_inference_steps=steps, 
            mc_algo='mc',
            guidance_scale=guidance_scale,
            octree_resolution=octree_resolution,
            generator=torch.manual_seed(seed))[0]
        
        log.info(f"Generated mesh with {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        pipeline.to(offload_device)
        
        return (mesh, )
    
class Hy3DPostprocessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("HY3DMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, remove_floaters, remove_degenerate_faces, reduce_faces, max_facenum):
        if remove_floaters:
            mesh = FloaterRemover()(mesh)
            log.info(f"Removed floaters, resulting in {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")
        if remove_degenerate_faces:
            mesh = DegenerateFaceRemover()(mesh)
            log.info(f"Removed degenerate faces, resulting in {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")
        if reduce_faces:
            mesh = FaceReducer()(mesh, max_facenum=max_facenum)
            log.info(f"Reduced faces, resulting in {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces")
        
        return (mesh, )
    
class Hy3DExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("HY3DMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, mesh, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.glb')
        output_glb_path.parent.mkdir(exist_ok=True)
        mesh.export(output_glb_path)

        relative_path = Path(subfolder) / f'{filename}_{counter:05}_.glb'
        
        return (str(relative_path), )

NODE_CLASS_MAPPINGS = {
    "Hy3DModelLoader": Hy3DModelLoader,
    "Hy3DGenerateMesh": Hy3DGenerateMesh,
    "Hy3DExportMesh": Hy3DExportMesh,
    "DownloadAndLoadHy3DDelightModel": DownloadAndLoadHy3DDelightModel,
    "DownloadAndLoadHy3DPaintModel": DownloadAndLoadHy3DPaintModel,
    "Hy3DDelightImage": Hy3DDelightImage,
    "Hy3DRenderMultiView": Hy3DRenderMultiView,
    "Hy3DBakeFromMultiview": Hy3DBakeFromMultiview,
    "Hy3DTorchCompileSettings": Hy3DTorchCompileSettings,
    "Hy3DPostprocessMesh": Hy3DPostprocessMesh,
    "LoadCustomMesh": LoadCustomMesh,
    "Hy3DCameraConfig": Hy3DCameraConfig,
    "Hy3DMeshUVWrap": Hy3DMeshUVWrap,
    "Hy3DSampleMultiView": Hy3DSampleMultiView,
    "Hy3DMeshVerticeInpaintTexture": Hy3DMeshVerticeInpaintTexture,
    "Hy3DApplyTexture": Hy3DApplyTexture,
    "CV2InpaintTexture": CV2InpaintTexture,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DModelLoader": "Hy3DModelLoader",
    "Hy3DGenerateMesh": "Hy3DGenerateMesh",
    "Hy3DExportMesh": "Hy3DExportMesh",
    "DownloadAndLoadHy3DDelightModel": "(Down)Load Hy3D DelightModel",
    "DownloadAndLoadHy3DPaintModel": "(Down)Load Hy3D PaintModel",
    "Hy3DDelightImage": "Hy3DDelightImage",
    "Hy3DRenderMultiView": "Hy3D Render MultiView",
    "Hy3DBakeFromMultiview": "Hy3D Bake From Multiview",
    "Hy3DTorchCompileSettings": "Hy3D Torch Compile Settings",
    "Hy3DPostprocessMesh": "Hy3D Postprocess Mesh",
    "LoadCustomMesh": "Load Custom Mesh",
    "Hy3DCameraConfig": "Hy3D Camera Config",
    "Hy3DMeshUVWrap": "Hy3D Mesh UV Wrap",
    "Hy3DSampleMultiView": "Hy3D Sample MultiView",
    "Hy3DMeshVerticeInpaintTexture": "Hy3D Mesh Vertice Inpaint Texture",
    "Hy3DApplyTexture": "Hy3D Apply Texture",
    "CV2InpaintTexture": "CV2 Inpaint Texture",
    }
