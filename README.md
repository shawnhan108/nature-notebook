# nature-notebook
A set of notebooks that leverages classical ML algorithms and deep learning approaches to address a series of issues in the field of conservation and biology.

* CycleGAN Image Conversion: Cycle-consistent Generative Adversarial Network (CycleGAN), Keras.
* The Nature Conservancy Fisheries Monitoring: CNN, Keras.

## CycleGAN Image Conversion
### Model
A CycleGAN implemented in Keras and trained to take winter landscape photos as input, and output summer-styled version of the landscapes, and vice versa. The model is built according to the 2017 research paper [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593). Both discriminators are implemented as PatchGAN discriminators, and the generators are encoder-decoder models that uses ResNet blocks for interpretation. The generator [AtoB](https://drive.google.com/file/d/1XFilejrifw9C-lapk-d1RVxEuuWgjQQm/view?usp=sharing) and [BtoA](https://drive.google.com/file/d/1w9tvC_XjQRpA5rTOGoVo-jgT2y62c1gY/view?usp=sharing) can be access via the URL.

### Training
The model is trained on approximately 2000 images, about 1000 images for each generator. Every photo sample is resized to 256x256, and the network is trained on a Tesla V100 GPU.


## The Nature Conservancy Fisheries Monitoring
### Model
Two CNN models, one using an Adam optimizer and the other using a customized SGD optimizer, are implemented in Keras and trained to predict and classify the images of species of commercial fishery catches from a set of 8 classes, based on the images recorded by monitoring cameras on fishing ships provided by The Nature Conservancy. The model using Adam optimizer achives a logloss of 0.49 and an accuracy of 92% on the validating dataset.

### Training
The model is trained on a [dataset](https://drive.google.com/file/d/1liKRmKdbabq5NZqaLYr7FZEq6BVBAIBS/view?usp=sharing) of 8 classes, a total of approximately 3800 images provided by the Nature Conservancy. The training time is approximately 5.5 hours on an NVidia K80 GPU.

### Weights 
The Weights of the Models using an [Adam optimizer](https://drive.google.com/file/d/1dP1LCAm-hjWczF3joq5IqGQBJ5vWILTv/view?usp=sharing) and a [customized SGD](https://drive.google.com/file/d/1xbpDiAztdq7PEK6fG3GDcmoRy1yknJDj/view?usp=sharing) optimizer can be access via the URLs, respectively.
