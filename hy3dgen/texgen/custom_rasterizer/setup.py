from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

torch_version = torch.__version__.split('+')[0].replace('.', '')
cuda_version = torch.version.cuda.replace('.', '')

version = f"0.1.0+torch{torch_version}.cuda{cuda_version}"
# build custom rasterizer
# build with `python setup.py install`
# nvcc is needed

custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
])

setup(
    packages=find_packages(),
    version=version,
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    },   
)
