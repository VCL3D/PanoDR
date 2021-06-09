# PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes.

[![Paper](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/abs/2106.00446)
[![Conference](http://img.shields.io/badge/CVPR-2021-blue.svg?style=plastic)](http://cvpr2021.thecvf.com/)
[![Workshop](http://img.shields.io/badge/OmniCV-2021-lightblue.svg?style=plastic)](https://sites.google.com/view/omnicv2021/home)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/PanoDR/)

![](https://github.com/VCL3D/PanoDR/tree/gh-pages/assets/model.png?raw=true) 

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

## Training 

More info regarding the training of the model will be available soon!

## Inference

You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1TD0wJe4EncunD-ZiQ9RTQVXbIv-1Snz6?usp=sharing)
and specify the arguments `--eval_chkpnt_folder` and `--segmentation_model_chkpnt`, respectively.
Assuming the input image and mask are in the format as in the `input` folder run: 

```bash
python src/train/test.py --inference --eval_path input/
```


# Acknowledgements

This project has received funding from the European Union's Horizon 2020 innovation programme [ATLANTIS](https://atlantis-ar.eu) under grant agreement No 951900.

Our code borrows from [SEAN](https://github.com/ZPdesu/SEAN) and [deepfillv2](https://github.com/zhaoyuzhi/deepfillv2).
