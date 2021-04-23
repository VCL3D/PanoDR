# PanoDR

# Abstract
<img src="./assets/teaser.png" width="650"  title="Teaser" alt="Inpainted with scenes' layout annotated." align="right"/>

The rising availability of commercial 360o cameras that democratize indoor scanning, has increased the interest for novel applications, such as interior space re-design. Diminished Reality (DR) fulfills the requirement of such applications, to remove existing objects in the scene, essentially translating this to a counterfactual inpainting task. While recent advances in data-driven inpainting have shown significant progress in generating realistic samples, they are not constrained to produce results with reality mapped
structures. To preserve the ‘reality’ in indoor (re-)planning applications, the scene’s structure preservation is crucial. To ensure structure-aware counterfactual inpainting, we propose a model that initially predicts the structure of a indoor scene and then uses it to guide the reconstruction of an empty – background only – representation of the same scene. We train and compare against other state-of-the-art methods on a version of the Structured3D dataset [47] modified for DR, showing superior results in both quantitative metrics and qualitative results, but more interestingly, our approach exhibits a much faster convergence rate.




# Acknowledgements
This project has received funding from the European Union's Horizon 2020 innovation programme ATLANTIS under grant agreement No 951900.
