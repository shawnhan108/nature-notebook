# nature-notebook
A set of notebooks that leverage classical ML algorithms and deep learning approaches to address a series of issues in the field of conservation and biology.

* CycleGAN Image Conversion: Cycle-consistent Generative Adversarial Network (CycleGAN), generative, Keras.
* The Nature Conservancy Fisheries Monitoring: CNN, image classification, Keras.
* Sequential Protein Subcellular Localization Classification: RNN, LSTM, CNN, classification, Theano.
* Human Protein Atlas Image Classification: CNN, Inception-ResNet-V2, Vgg16, image-classification, Keras.

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
The weights of the Models using an [Adam optimizer](https://drive.google.com/file/d/1dP1LCAm-hjWczF3joq5IqGQBJ5vWILTv/view?usp=sharing) and a [customized SGD](https://drive.google.com/file/d/1xbpDiAztdq7PEK6fG3GDcmoRy1yknJDj/view?usp=sharing) optimizer can be access via the URLs, respectively.


## Sequential Protein Subcellular Localization Classification
### Model
A Recurrent Neural Network (RNN), specifically a CNN-LSTM model is implemented in Theano to single-class classify the subcellular localization of a protein given its sequence, from a set of 10 classes of subcellular structures. The model consists of both convolutional layers and a pair of bidirectional LSTM (RNN) layers. The model achieves around 80% - 85% accuracy.

### Training
The model is trained on the *MultiLoc* dataset, which consists of around 6000 protein sequence samples. Each protein sequence is labelled with a class of subcellular structure that it localizes. The model is trained on AMD Radeon Pro 5300M CPU.

## Human Protein Atlas Image Classification
### Model
Two CNN models, one using the pre-trained imagenet Inception-Resnet-V2 model and the other using the pretrained Vgg-16 model from Keras applications library, are implemented in Keras and trained to predict and multi-label classify the protein's subcellular localization from a set of 28 cell organelle classes. The classification is based on the confocal microscopy images of proteins provided by the [Human Protein Atlas](https://www.proteinatlas.org/) program. The Inception-Resnet-V2 model achieves an F1-score of 0.6379 and thus out-performs the vgg-model (with an F1-score of 0.5043).

### Training
The model is trained on an approximately 14GB [dataset](https://www.kaggle.com/c/human-protein-atlas-image-classification/data) provided by the *Human Protein Atlas* program, including more than 124,000 confocal microscopy images of human proteins. The training time is approximately 2 hours for the Inception-Resnet-V2 model and 1.5 hours for the Vgg16 model on an NVidia K80 GPU.

### Checkpoints
The checkpoints of the Models using the [Inception-Resnet-V2 model](https://drive.google.com/file/d/1saABuaSbW_-nGEQqzcLeO36s6aES10xB/view?usp=sharing) and the [Vgg16 model](https://drive.google.com/file/d/1zyAwpOU82lppnEsR6oJd2Jx0FqBuBZxU/view?usp=sharing) can be access via the URLs, respectively.

### 
