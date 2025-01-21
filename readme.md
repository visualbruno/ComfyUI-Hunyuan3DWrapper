# ComfyUI wrapper for [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)

# WORKINPROGRESS
# installation still messy, requires compiling for texture gen

Main model, original: https://huggingface.co/tencent/Hunyuan3D-2/blob/main/hunyuan3d-dit-v2-0/model.ckpt

Converted to .safetensors: https://huggingface.co/Kijai/Hunyuan3D-2_safetensors

to `ComfyUI/diffusion_models/`

Rest of the models are diffusers models, so they are wrapped and autodownloaded for now.

# Installation
Dependencies:
```
pip install -r requirements.txt

portable:

`python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\requirements.txt`
```

For the texturegen part compilation is needed, I have included my compilations as a wheel for the rasterizer, and compiled .pyd for the mesh_processor (already in place), these are compiled for:

- Windows 11 python 3.12 cu126 (works with torch build on 124)

You would do `pip install wheels\custom_rasterizer-0.1-cp312-cp312-win_amd64.whl`

or with portable (in `ComfyUI_windows_portable` -folder):
`python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\wheels\custom_rasterizer-0.1-cp312-cp312-win_amd64.whl` 

This was tested to work on latest ComfyUI portable install

For anything else you need to compile yourself:

```
cd hy3dgen/texgen/custom_rasterizer
python setup.py install

cd hy3dgen/texgen/differentiable_renderer
python setup.py build_ext --inplace
```

![alt text](image.png)