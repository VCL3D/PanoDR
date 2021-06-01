# PanoDR: Spherical Panorama Diminished Reality for Indoor Scenes

<img src="./assets/teaser.png" width="1000"  title="Teaser" alt="Inpainted with scenes' layout annotated." align="center"/>

>Diminishing the highlighted (<span style="color:red">red</span> mask) object in indoor spherical panorama images. White lines annotate the
scene’s layout in panorama and perspective views. Left to right: i) masked object to remove, ii) pure inpainting result of
state-of-the-art methods (top row: RFR, bottom row: PICNet), iii) perspective view of inpainted region by these
methods better shows that they do not necessarily respect the scene’s structural layout, iv) our panorama inpainting that takes
a step towards preserving the structural reality, v) perspective view of inpainted region by our model, showing superior results
both in texture generation and layout preservation. The results in this figure depict cases where RFR and PICNet provide
reasonable structural coherency and aim at showcasing our model’s finer-grained accuracy.

# Abstract
The rising availability of commercial 360° cameras that democratize indoor scanning, has increased the interest for novel applications, such as interior space re-design. Diminished Reality (DR) fulfills the requirement of such applications, to remove existing objects in the scene, essentially translating this to a counterfactual inpainting task. While recent advances in data-driven inpainting have shown significant progress in generating realistic samples, they are not constrained to produce results with reality mapped
structures. To preserve the ‘reality’ in indoor (re-)planning applications, the scene’s structure preservation is crucial. To ensure structure-aware counterfactual inpainting, we propose a model that initially predicts the structure of a indoor scene and then uses it to guide the reconstruction of an empty – background only – representation of the same scene. We train and compare against other state-of-the-art methods on a version of the Structured3D dataset [1] modified for DR, showing superior results in both quantitative metrics and qualitative results, but more interestingly, our approach exhibits a much faster convergence rate.

#  PanoDR Model Architecture
<img src="./assets/model.png" width="1200"  title="Teaser" alt="Inpainted with scenes' layout annotated." align="center"/>

>The input masked image is encoded twice, once densely by the structure UNet encoder outputing a layout segmentation map,
and once by the surrounding context encoder, capturing the scene’s context while taking the mask into account via series
gated convolutions. These are then combined by the structure-aware decoder with a set of per layout component style
codes that are extracted by the complete input image. Two SEAN residual blocks ensure the structural alignment of the
reconstructed background image that is supervised by low- and high-level losses, as well as an adversarial loss driven by the
background image discriminator. The final diminished result is created via compositing the predicted and input images using
the diminishing mask.


# Qualitative results for diminishing objects from scenes in our test set. 
<p float="left">
<img src="./assets/qualitative/scene_03362_2D_rendering_13188_panorama_masked_inv_mask_orig-min.png" width="24%" title="03362_13188"/>
<img src="./assets/qualitative/scene_03362_13188_rfr.png" width="24%" title="03362_13188_Ours"/>
<img src="./assets/qualitative/scene_03362_2D_rendering_13188_panorama_PIC.png" width="24%" title="03362_13188_Ours"/>
<img src="./assets/qualitative/scene_03362_2D_rendering_13188_panorama_ours.png" width="24%" title="03362_13188_Ours"/>
</p>
<p float="left">
<img src="./assets/qualitative/scene_03498_2D_rendering_274746_panorama_masked_inv_mask_orig.png" width="24%" title="03498_274746"/>
<img src="./assets/qualitative/scene_03498_2D_rendering_274746_panorama_PIC.png" width="24%" title="03498_274746_RFR"/>
<img src="./assets/qualitative/scene_03498_2D_rendering_274746_panorama_rfr.png" width="24%" title="03498_274746_PICNet"/>
<img src="./assets/qualitative/scene_03498_2D_rendering_274746_panorama_ours.png" width="24%" title="03498_274746_Ours"/>
</p>
<p float="left">
<img src="./assets/qualitative/scene_03312_2D_rendering_203_panorama_masked_inv_mask-min.png" width="24%" title="03312_203"/>
<img src="./assets/qualitative/scene_03312_2D_rendering_203_panorama_out_0_PIC.png" width="24%" title="03312_203_RFR"/>
<img src="./assets/qualitative/scene_03312_203_rfr.png" width="24%" title="03312_203_PICNet"/>
<img src="./assets/qualitative/scene_03312_2D_rendering_203_panorama_ours-min.png" width="24%" title="03312_203_Ours"/>
</p>

<p float="left">
<img src="./assets/qualitative/scene_03424_2D_rendering_947433_panorama_masked_inv_mask.png" width="24%" title="03498_13188"/>
<img src="./assets/qualitative/scene_03424_947433_rfr.png" width="24%" title="03424_947433_RFR"/>
<img src="./assets/qualitative/scene_03424_2D_rendering_947433_panorama_out_0_PIC.png" width="24%" title="03424_947433_PICNet"/>
<img src="./assets/qualitative/scene_03424_2D_rendering_947433_panorama_ours.png" width="24%" title="03424_947433_Ours"/>
</p>

<p float="left">
<img src="./assets/qualitative/scene_03316_2D_rendering_652088_panorama_masked_inv_mask.png" width="24%" title="03316_652088"/>
<img src="./assets/qualitative/scene_03316_2D_rendering_652088_panorama_masked_RFR-min.png" width="24%" title="03316_652088_RFR"/>
<img src="./assets/qualitative/scene_03316_2D_rendering_652088_panorama_PIC.png" width="24%" title="03316_652088_PICNet"/>
<img src="./assets/qualitative/scene_03316_2D_rendering_652088_panorama_ours.png" width="24%" title="03316_652088_Ours"/>
</p>

<p float="left">
<img src="./assets/qualitative/scene_03461_2D_rendering_421_panorama_masked_inv_mask.png" width="24%" title="03461_421"/>
<img src="./assets/qualitative/scene_03461_421_RFR.png" width="24%" title="03461_421_RFR"/>
<img src="./assets/qualitative/scene_03461_2D_rendering_421_panorama_PIC.png" width="24%" title="03461_421_PICNet"/>
<img src="./assets/qualitative/scene_03461_2D_rendering_421_panorama_ours.png" width="24%" title="03461_421_Ours"/>
</p>

>From Left to right: i) Input image with the diminished area masked with transparent red, ii) RFR, iii) PICNet and iv) ours.

# AR Application

<p float="left">
<img src="./assets/ar_dr/original.png" width="24%" title="03461_421"/>
<img src="./assets/ar_dr/ar_no_dr.png" width="24%" title="03461_421_RFR"/>
<img src="./assets/ar_dr/masked.png" width="24%" title="03461_421_PICNet"/>
<img src="./assets/ar_dr/ar.png" width="24%" title="03461_421_Ours"/>
</p>

>A demonstration of our method, which levitates AR/DR applications. From left to right: i) original panorama ii) augmented reality
without diminished reality iii) highlighted object for removal iv) our inpainting method result v) augmented reality with
diminished reality, the result is much more natural.

# Contact
Please direct any questions related to the code & models to gkitsasv “at” iti “dot” gr or post an issue to the code [repo](https://github.com/VCL3D/PanoDR).

# Acknowledgements
This project has received funding from the European Union's Horizon 2020 innovation programme [ATLANTIS](https://atlantis-ar.eu) under grant agreement No 951900.

<p float="left">
<img src="./assets/eu.png" width="48%" title="eu"/>
<img src="./assets/atlantis_logo.png" width="48%" height="75%" title="atlantis_logo"/>
</p>


# References
<a name="S3D"/>__\[1\]__ Zheng, J., Zhang, J., Li, J., Tang, R., Gao, S. and Zhou, Z., 2020. [Structured3d: A large photo-realistic dataset for structured 3d modeling.](https://arxiv.org/pdf/1908.00222.pdf) In 16th European Conference on Computer Vision, ECCV 2020 (pp. 519-535). Springer Science and Business Media Deutschland GmbH.
