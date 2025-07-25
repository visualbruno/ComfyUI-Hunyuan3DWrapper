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

import copy
import importlib
import inspect
import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from comfy.utils import ProgressBar, load_torch_file
import comfy.model_management as mm

logger = logging.getLogger(__name__)

from .schedulers import FlowMatchEulerDiscreteScheduler, ConsistencyFlowMatchEulerDiscreteScheduler

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    try:
        obj = getattr(importlib.import_module(module, package=os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), cls)
    except:
        obj = getattr(importlib.import_module(module, package=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath( __file__ ))))), cls)
    return obj


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    cls = get_obj_from_str(config["target"])
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    @classmethod
    def from_single_file(
        cls,
        ckpt_path,
        device='cuda',
        offload_device=torch.device('cpu'),
        dtype=torch.float16,
        use_safetensors=None,
        compile_args=None,
        attention_mode="sdpa",
        cublas_ops=False,
        scheduler="FlowMatchEulerDiscreteScheduler", 
        **kwargs,
    ):
        new_sd = {}
        sd = load_torch_file(ckpt_path)
        if ckpt_path.endswith('.safetensors'):
            for key, value in sd.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in new_sd:
                    new_sd[model_name] = {}
                new_sd[model_name][new_key] = value

        script_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # load config

        single_block_nums = set()
        for k in new_sd["model"].keys():
            if k.startswith('single_blocks.'):
                block_num = int(k.split('.')[1])
                single_block_nums.add(block_num)
    
        if len(single_block_nums) < 17:
            config_path = os.path.join(script_directory, "configs", "dit_config_mini.yaml")
            logger.info(f"Model has {len(single_block_nums)} single blocks, setting config to dit_config_mini.yaml")
        else:
            config_path = os.path.join(script_directory, "configs", "dit_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        
        # load model
        if "guidance_in.in_layer.bias" in new_sd['model']: #guidance_in.in_layer.bias
            logger.info("Model has guidance_in, setting guidance_embed to True")
            config['model']['params']['guidance_embed'] = True
            config['conditioner']['params']['main_image_encoder']['kwargs']['has_guidance_embed'] = True
        config['model']['params']['attention_mode'] = attention_mode
        #config['vae']['params']['attention_mode'] = attention_mode

        #if cublas_ops:
        #    config['vae']['params']['cublas_ops'] = True
        
        with init_empty_weights():
            model = instantiate_from_config(config['model'])
            vae = instantiate_from_config(config['vae'])
            conditioner = instantiate_from_config(config['conditioner'])
        #model
        for name, param in model.named_parameters():
            set_module_tensor_to_device(model, name, device=offload_device, dtype=dtype, value=new_sd['model'][name])
        #vae
        for name, param in vae.named_parameters():
            set_module_tensor_to_device(vae, name, device=offload_device, dtype=dtype, value=new_sd['vae'][name])       
        
        if 'conditioner' in new_sd:
            #conditioner.load_state_dict(ckpt['conditioner'])
            for name, param in conditioner.named_parameters():
                set_module_tensor_to_device(conditioner, name, device=offload_device, dtype=dtype, value=new_sd['conditioner'][name])

        image_processor = instantiate_from_config(config['image_processor'])

        if scheduler == "FlowMatchEulerDiscreteScheduler":
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        elif scheduler == "ConsistencyFlowMatchEulerDiscreteScheduler":
            scheduler = ConsistencyFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, pcm_timesteps=100)
        
        #scheduler = instantiate_from_config(config['scheduler'])

        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            if compile_args["compile_transformer"]:
                model = torch.compile(model)
            if compile_args["compile_vae"]:
                vae = torch.compile(vae)

        model_kwargs = dict(
            #vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            offload_device=offload_device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)

        return cls(**model_kwargs), vae

    def __init__(
        self,
        #vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device=torch.device('cuda'),
        offload_device=torch.device('cpu'),
        dtype=torch.float16,
        **kwargs
    ):
        #self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor

        self.main_device = device
        self.offload_device = offload_device

        self.to(offload_device, dtype)

    def to(self, device=None, dtype=None):
        if device is not None:
            #self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)
        if dtype is not None:
            self.dtype = dtype
            #self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)

    def encode_cond(self, image, mask, do_classifier_free_guidance, dual_guidance, view_dict=None):
        self.conditioner.to(self.main_device)
        bsz = 1
        cond = self.conditioner(image=image, mask=mask, view_dict=view_dict)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                un_cond = self.conditioner.unconditional_embedding(bsz)

                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        self.conditioner.to(self.offload_device)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        #shape = (batch_size, *self.vae.latent_shape)
        num_latents = 3072
        embed_dim = 64
        shape = (batch_size, num_latents, embed_dim)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image):
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]
        image_pts = []
        mask_pts = []
        for img in image:
            image_pt, mask_pt = self.image_processor(img, return_mask=True)
            image_pts.append(image_pt)
            mask_pts.append(mask_pt)

        image_pts = torch.cat(image_pts, dim=0).to(self.main_device, dtype=self.dtype)
        if mask_pts[0] is not None:
            mask_pts = torch.cat(mask_pts, dim=0).to(self.main_device, dtype=self.dtype)
        else:
            mask_pts = None
        return image_pts, mask_pts

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    # @torch.no_grad()
    # def __call__(
    #     self,
    #     image: Union[str, List[str], Image.Image] = None,
    #     num_inference_steps: int = 50,
    #     timesteps: List[int] = None,
    #     sigmas: List[float] = None,
    #     eta: float = 0.0,
    #     guidance_scale: float = 7.5,
    #     dual_guidance_scale: float = 10.5,
    #     dual_guidance: bool = True,
    #     generator=None,
    #     box_v=1.01,
    #     octree_resolution=384,
    #     mc_level=-1 / 512,
    #     num_chunks=8000,
    #     mc_algo='mc',
    #     output_type: Optional[str] = "trimesh",
    #     enable_pbar=True,
    #     **kwargs,
    # ) -> List[List[trimesh.Trimesh]]:
    #     callback = kwargs.pop("callback", None)
    #     callback_steps = kwargs.pop("callback_steps", None)

    #     device = self.main_device
    #     dtype = self.dtype
    #     do_classifier_free_guidance = guidance_scale >= 0 and \
    #                                   getattr(self.model, 'guidance_cond_proj_dim', None) is None
    #     dual_guidance = dual_guidance_scale >= 0 and dual_guidance

    #     image, mask = self.prepare_image(image)
    #     cond = self.encode_cond(image=image,
    #                             mask=mask,
    #                             do_classifier_free_guidance=do_classifier_free_guidance,
    #                             dual_guidance=dual_guidance)
    #     batch_size = image.shape[0]

    #     t_dtype = torch.long
    #     timesteps, num_inference_steps = retrieve_timesteps(
    #         self.scheduler, num_inference_steps, device, timesteps, sigmas)

    #     latents = self.prepare_latents(batch_size, dtype, device, generator)
    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    #     guidance_cond = None
    #     if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
    #         print('Using lcm guidance scale')
    #         guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
    #         guidance_cond = self.get_guidance_scale_embedding(
    #             guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
    #         ).to(device=device, dtype=latents.dtype)

    #     comfy_pbar = ProgressBar(num_inference_steps)

    #     self.model.to(device)
    #     for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
    #         # expand the latents if we are doing classifier free guidance
    #         if do_classifier_free_guidance:
    #             latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
    #         else:
    #             latent_model_input = latents
    #         latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

    #         # predict the noise residual
    #         timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
    #         timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
    #         noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

    #         # no drop, drop clip, all drop
    #         if do_classifier_free_guidance:
    #             if dual_guidance:
    #                 noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
    #                 noise_pred = (
    #                     noise_pred_uncond
    #                     + guidance_scale * (noise_pred_clip - noise_pred_dino)
    #                     + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
    #                 )
    #             else:
    #                 noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
    #                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    #         # compute the previous noisy sample x_t -> x_t-1
    #         outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
    #         latents = outputs.prev_sample

    #         comfy_pbar.update(1)

    #         if callback is not None and i % callback_steps == 0:
    #             step_idx = i // getattr(self.scheduler, "order", 1)
    #             callback(step_idx, t, outputs)
    #     self.model.to(self.offload_device)
    #     mm.soft_empty_cache()

    #     return self._export(
    #         latents,
    #         output_type,
    #         box_v, mc_level, num_chunks, octree_resolution, mc_algo,
    #     )

    # def _export(self, latents, output_type, box_v, mc_level, num_chunks, octree_resolution, mc_algo):
    #     if not output_type == "latent":
    #         self.vae.to(self.main_device)
    #         latents = 1. / self.vae.scale_factor * latents
    #         latents = self.vae(latents)
    #         outputs = self.vae.latents2mesh(
    #             latents,
    #             bounds=box_v,
    #             mc_level=mc_level,
    #             num_chunks=num_chunks,
    #             octree_resolution=octree_resolution,
    #             mc_algo=mc_algo,
    #         )
    #         self.vae.to(self.offload_device)
    #     else:
    #         outputs = latents

    #     if output_type == 'trimesh':
    #         outputs = export_to_trimesh(outputs)

    #     return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        #eta: float = 0.0,
        guidance_scale: float = 7.5,
        generator=None,
        # box_v=1.01,
        # octree_resolution=384,
        # mc_level=0.0,
        # mc_algo='mc',
        # num_chunks=8000,
        # output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        view_dict=None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        device = self.main_device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        #image, mask = self.prepare_image(image)

        cond = self.encode_cond(
            image=image,
            mask=mask,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
            view_dict=view_dict
        )
        batch_size = 1

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
        print("guidance: ", guidance)

        comfy_pbar = ProgressBar(num_inference_steps)
        for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
            # expand the latents if we are doing classifier free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            # NOTE: we assume model get timesteps ranged from 0 to 1
            timestep = t.expand(latent_model_input.shape[0]).to(
                latents.dtype) / self.scheduler.config.num_train_timesteps
            noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

            if do_classifier_free_guidance:
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            outputs = self.scheduler.step(noise_pred, t, latents)
            latents = outputs.prev_sample

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, outputs)
            comfy_pbar.update(1)
        print("latents shape: ", latents.shape)
        return latents
        # return self._export(
        #     latents,
        #     output_type,
        #     box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        # )
