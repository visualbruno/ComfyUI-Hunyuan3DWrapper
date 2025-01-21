# ComfyUI wrapper for [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)

# WORKINPROGRESS
# installation still messy, requires compiling for texture gen

Main model, original: https://huggingface.co/tencent/Hunyuan3D-2/blob/main/hunyuan3d-dit-v2-0/model.ckpt

Converted to .safetensors: https://huggingface.co/Kijai/Hunyuan3D-2_safetensors

to `ComfyUI/diffusion_models/`

Rest of the models are diffusers models, so they are wrapped and autodownloaded for now.



```
pip install -r requirements.txt
# for texture (Linux only for now)
cd hy3dgen/texgen/custom_rasterizer
python3 setup.py install
cd hy3dgen/texgen/differentiable_renderer
bash compile_mesh_painter.sh
```

![alt text](image.png)