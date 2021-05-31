# PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes.

[![Conference](http://img.shields.io/badge/CVPR-2021-blue.svg?style=plastic)](http://cvpr2021.thecvf.com/)
[![Workshop](http://img.shields.io/badge/OmniCV-2021-lightblue.svg?style=plastic)](https://sites.google.com/view/omnicv2021/home)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/PanoDR/)

## Prerequisites
- Windows10 or Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Installation
- Clone this repo:

```bash
git clone https://github.com/VCL3D/PanoDR.git
cd PanoDR
```

## Inference

You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1TD0wJe4EncunD-ZiQ9RTQVXbIv-1Snz6?usp=sharing)
and specify the `--eval_chkpnt_folder` and `--segmentation_model_chkpnt`, respectively.
Assuming the input image and mask are in the format as in the `input` folder run: 

`python src/train/test.py --inference --eval_path input/`


