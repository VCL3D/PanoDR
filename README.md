# PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes.

[![Paper](http://img.shields.io/badge/paper-arxiv-critical.svg?style=plastic)](https://arxiv.org/abs/2106.00446)
[![Conference](http://img.shields.io/badge/CVPR-2021-blue.svg?style=plastic)](http://cvpr2021.thecvf.com/)
[![Workshop](http://img.shields.io/badge/OmniCV-2021-lightblue.svg?style=plastic)](https://sites.google.com/view/omnicv2021/home)
[![YouTube](https://img.shields.io/badge/Presentation-YouTube-red.svg?style=plastic)](https://www.youtube.com/watch?v=xa7Fl2mD4CA&t=26274s)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/PanoDR/)  <br />


# Model Architecture <br />
![](https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/PanoDR_Model.png) <br />

 

## Prerequisites
- Windows10 or Linux
- Python 3.7
- NVIDIA GPU + CUDA CuDNN
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
pip install -e .
```

## Dataset

We use [Structured3D](https://structured3d-dataset.org/) dataset. To train a model on the dataset please download the dataset from the official website. We follow the official training, validation, and testing splits as defined by the authors. After downloading the dataset, split the scenes into training train, validation and test folders. The folders should have the following format:

```
Structured3D/
    train/
        scene_00000
        scene_00001
        ...
    test/
        scene_03250
        scene_03251
        ...
    validation/
        scene_03000
        scene_03001
        ...
```
In order to estimate the dense layout maps, specify the path to train and test folders and run:

```bash
python src\utils\vcl3datlantis\dataset\precompute_structure_semantics.py 
```



[Structured3D](https://structured3d-dataset.org/) provides 3 different room configurations, __empty__, __simple__ and __full__, which,in theory, enables this dataset for Diminished Reality applications. In practice, this statement doesn't hold since the rooms are rendered with ray-tracing, fact that affects the global illumination and texture, so replacing a region from the __full__ configuration with the corresponding __empty__ one does not create photo-consistent results.
<table align='center'>
<tr>
 <td><img src='https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/dataset/scene_03402_521088_masked_highlighted.png' width='512' height='256'/></td>
 <td><img src='https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/dataset/scene_03402_521088_masked_erroneous.png' width='512' height='256'/></td>
</tr>
<tr>
 <td>Full room configuration with annotated the object for diminsing</td>
 <td>Full room configuration with replaced region from empty room configuration</td>
</tr>
</table>

It is obvious that the diminished scene has large photometric insconsistency at the diminished region, which is not suitable for deep learning algorithms. To overcome this barrier, we start to _augment_ the __empty__ rooms with objects from the corresponding __full__ room configurations, so that replacing a region is photo-consistent.
<table align='center'>
<tr>
 <td><img src='https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/dataset/scene_03402_521088.png' width='512' height='256'/></td>
 <td><img src='https://github.com/VCL3D/PanoDR/blob/gh-pages/assets/dataset/scene_03402_521088_masked_correct.png' width='512' height='256'/></td>
</tr>
<tr>
 <td>Empty room configuration with augmented objects</td>
 <td>Augmented room with diminished table</td>
</tr>
</table>
The only issue of this approach of creating samples is the abscence of shadows, which makes the whole scene less realistic, but still the gain of this method is greater for Diminished Reality applications.

## Training 

More info regarding the training of the model will be available soon!

## Inference

You can download the pre-trained models from [here](https://drive.google.com/drive/folders/1TD0wJe4EncunD-ZiQ9RTQVXbIv-1Snz6?usp=sharing)
and specify the arguments `--eval_chkpnt_folder` and `--segmentation_model_chkpnt`, respectively.
Assuming the input image and mask are in the format as in the `input` folder run: 

```bash
python src/train/test.py --inference --eval_path input/
```

## Citation
If you use this code for your research, please cite the following:
```
@inproceedings{gkitsas2021panodr,
  title={PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes},
  author={Gkitsas, Vasileios and Sterzentsenko, Vladimiros and Zioulis, Nikolaos and Albanis, Georgios and Zarpalas, Dimitrios},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3716--3726},
  year={2021}
}
```


# Acknowledgements

This project has received funding from the European Union's Horizon 2020 innovation programme [ATLANTIS](https://atlantis-ar.eu) under grant agreement No 951900.

Our code borrows from [SEAN](https://github.com/ZPdesu/SEAN) and [deepfillv2](https://github.com/zhaoyuzhi/deepfillv2).
