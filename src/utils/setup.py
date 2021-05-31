from setuptools import setup, find_packages

NAME = 'vcl3datlantis'
DESCRIPTION = 'PyTorch module VCL project ATLANTIS'
AUTHOR = 'VCL3D'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '1.3'

REQUIREMENTS = [
    'numpy',
    'opencv-python',
    'scikit-image',
    'pillow',
    'pyyaml',
    'pycocotools',
    'timm==0.1.20',
    'efficientnet_pytorch',
    'pretrainedmodels',
    'visdom',
    'pytorch-lightning',
    'imageio',
    'unique_color'
]

setup(
    name=NAME,
    description=DESCRIPTION,
    version=VERSION,
    author=AUTHOR,
    install_requires = REQUIREMENTS,
    packages = ['vcl3datlantis']
)