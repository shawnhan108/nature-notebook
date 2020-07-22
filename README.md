# nature-notebook
A set of notebooks that leverages classical ML algorithms and deep learning approaches to address a series of issues in the field of conservation and biology.

* CycleGAN Image Conversion: Cycle-consistent Generative Adversarial Network (CycleGAN).

## CycleGAN Image Conversion
### Model
A CycleGAN implemented in Keras and trained to take winter landscape photos as input, and output summer-styled version of the landscapes, and vice versa. The model is built according to the 2017 research paper [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593). Both discriminators are implemented as PatchGAN discriminators, and the generators are encoder-decoder models that uses ResNet blocks for interpretation. The generator [AtoB](https://drive.google.com/file/d/1XFilejrifw9C-lapk-d1RVxEuuWgjQQm/view?usp=sharing) and [BtoA](https://drive.google.com/file/d/1w9tvC_XjQRpA5rTOGoVo-jgT2y62c1gY/view?usp=sharing) can be access via the links.

### Training
The model is trained on approximately 2000 images, about 1000 images for each generator. Every photo sample is resized to 256x256, and the network is trained on a Tesla V100 GPU.
