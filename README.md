# PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes.

[![Conference](http://img.shields.io/badge/CVPR-2021-blue.svg?style=plastic)](http://cvpr2021.thecvf.com/)
[![Workshop](http://img.shields.io/badge/OmniCV-2021-lightblue.svg?style=plastic)](https://sites.google.com/view/omnicv2021/home)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/PanoDR/)

## License

All rights reserved. Licensed under the [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Attribution-NonCommercial-NoDerivatives 4.0 International)

## Prerequisites
- Windows10 or Linux
- Python 3.7
- CPU or NVIDIA GPU + CUDA CuDNN
- PyTorch 1.7.1 (or higher)

## Installation
- Clone this repo:

```bash
git clone https://github.com/VCL3D/PanoDR.git
cd PanoDR
```

- We recommend setting up a virtual environment (follow the `virtualenv` documentation).
Once your environment is set up and activated, install the `vcl3datlantis` package:

```bash
cd src/utils
pip install -e.
```

## Dataset

We use [Structured3D](https://structured3d-dataset.org/) dataset. To train a model on the dataset please download the dataset from the official website. We follow the official training, validation, and testing splits as defined by the authors.

## Inference

You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1TD0wJe4EncunD-ZiQ9RTQVXbIv-1Snz6?usp=sharing)
and specify the arguments `--eval_chkpnt_folder` and `--segmentation_model_chkpnt`, respectively.
Assuming the input image and mask are in the format as in the `input` folder run: 

```bash
python src/train/test.py --inference --eval_path input/
```


