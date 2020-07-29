# nature-notebook
A set of notebooks that leverage classical ML algorithms and DL neural nets using TF, Keras and Theano to address a series of issues in the field of conservation and biology.

* [CycleGAN Image Conversion](#cyclegan-image-conversion): Cycle-consistent Generative Adversarial Network (CycleGAN), generative, Keras.
* [The Nature Conservancy Fisheries Monitoring](#the-nature-conservancy-fisheries-monitoring): CNN, image classification, Keras.
* [Sequential Protein Subcellular Localization Classification](#sequential-protein-subcellular-localization-classification): RNN, bidirectional LSTM, CNN, classification, Theano.
* [Human Protein Atlas Image Classification](#human-protein-atlas-image-classification): CNN, Inception-ResNet-V2, Vgg16, image-classification, Keras.
* [Forest Cover Type Analysis](#forest-cover-type-analysis): KNN, Gaussian Naïve Bayes, Decision Tree, SVM, Random Forest, Extra Trees, Boosts, ANN, feature-selection, classification, Tensorflow, Scikit-learn.
* [Australian Bushfire Analysis](#australian-bushfire-analysis): KNN, LR, Decision Tree, Random Forest, SVM/SVR, classification, regression, ANN, data visualization, Seaborn, Scikit-learn, Tensorflow.
* [Fathead Minnow Toxicity Analysis](#fathead-minnow-toxicity-analysis): SVR, LR, ANN, regression, Seaborn, Scikit-learn, Tensorflow.

## CycleGAN Image Conversion
### Model
A CycleGAN implemented in Keras and trained to take winter landscape photos as input, and output summer-styled version of the landscapes, and vice versa. The model is built according to the 2017 research paper [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593). Both discriminators are implemented as PatchGAN discriminators, and the generators are encoder-decoder models that uses ResNet blocks for interpretation. The generator [AtoB](https://drive.google.com/file/d/1XFilejrifw9C-lapk-d1RVxEuuWgjQQm/view?usp=sharing) and [BtoA](https://drive.google.com/file/d/1w9tvC_XjQRpA5rTOGoVo-jgT2y62c1gY/view?usp=sharing) can be accessed via the URL.

### Training
The model is trained on approximately 2000 images, about 1000 images for each generator. Every photo sample is resized to 256x256, and the network is trained on a Tesla V100 GPU.


## The Nature Conservancy Fisheries Monitoring
### Model
Two CNN models, one using an Adam optimizer and the other using a customized SGD optimizer, are implemented in Keras and trained to predict and classify the images of species of commercial fishery catches from a set of 8 classes, based on the images recorded by monitoring cameras on fishing ships provided by The Nature Conservancy. The model using Adam optimizer achives a logloss of 0.49 and an accuracy of 92% on the validating dataset.

### Training
The model is trained on a [dataset](https://drive.google.com/file/d/1liKRmKdbabq5NZqaLYr7FZEq6BVBAIBS/view?usp=sharing) of 8 classes, a total of approximately 3800 images provided by the Nature Conservancy. The training time is approximately 5.5 hours on an NVidia K80 GPU.

### Weights 
The weights of the Models using an [Adam optimizer](https://drive.google.com/file/d/1dP1LCAm-hjWczF3joq5IqGQBJ5vWILTv/view?usp=sharing) and a [customized SGD](https://drive.google.com/file/d/1xbpDiAztdq7PEK6fG3GDcmoRy1yknJDj/view?usp=sharing) optimizer can be accessed via the URLs, respectively.


## Sequential Protein Subcellular Localization Classification
### Model
A Recurrent Neural Network (RNN), specifically a CNN-LSTM model is implemented in Theano to single-class classify the subcellular localization of a protein given its sequence, from a set of 10 classes of subcellular structures. The model consists of both convolutional layers and a pair of bidirectional LSTM (RNN) layers. The model achieves around 80% - 85% accuracy.

### Training
The model is trained on the *MultiLoc* dataset, which consists of around 6000 protein sequence samples. Each protein sequence is labelled with a class of subcellular structure that it localizes. The model is trained on AMD Radeon Pro 5300M CPUs.

## Human Protein Atlas Image Classification
### Model
Two CNN models, one using the pre-trained imagenet Inception-Resnet-V2 model and the other using the pretrained Vgg-16 model from Keras applications library, are implemented in Keras and trained to predict and multi-label classify the protein's subcellular localization from a set of 28 cell organelle classes. The classification is based on the confocal microscopy images of proteins provided by the [Human Protein Atlas](https://www.proteinatlas.org/) program. The Inception-Resnet-V2 model achieves an F1-score of 0.6379 and thus out-performs the vgg-model (with an F1-score of 0.5043).

### Training
The model is trained on an approximately 14GB [dataset](https://www.kaggle.com/c/human-protein-atlas-image-classification/data) provided by the *Human Protein Atlas* program, including more than 124,000 confocal microscopy images of human proteins. The training time is approximately 2 hours for the Inception-Resnet-V2 model and 1.5 hours for the Vgg16 model on an NVidia K80 GPU.

### Checkpoints
The checkpoints of the Models using the [Inception-Resnet-V2 model](https://drive.google.com/file/d/1saABuaSbW_-nGEQqzcLeO36s6aES10xB/view?usp=sharing) and the [Vgg16 model](https://drive.google.com/file/d/1zyAwpOU82lppnEsR6oJd2Jx0FqBuBZxU/view?usp=sharing) can be accessed via the URLs, respectively.

## Forest Cover Type Analysis
### Purpose
Forest cover type is a basic but crucial geographical feature. Multiple feature extraction algorithms and statistical/ML classification models are attempted to predict the forest cover type of a location in Roosevelt National Forest, CO, based on cartographic information only.

### Feature Analysis
Classifiers and algorithms including Extra Tree, Gradient Boost, Random Forest, XGBoost, Recursive Feature Elimiation, and Select Percentile are implemented using Scikit-learn.

### Model
A variety of ML statistical models and an ANN model are implemented in Scikit-learn and Tensorflow in search of the best performing model to predict the forest cover type. These classification model includes Linear Discriminant, Logistic Regression, K Nearest Neighbours, Gaussian Naïve Bayes, Decision Tree, Support Vector Machine, Random Forest, Extra Trees, AdaBoost, Bagging, Voting, XGB Classifier, and an ANN feedforward model. It is eventually concluded that the Extra Trees Classifier is the best performing model with an accuracy of approximately 87%. This best-performing [Extra Trees Classifier model](https://drive.google.com/file/d/1OEKXNf_lfz2F1oWW0eSy9BIOT5eQreMt/view?usp=sharing) can be accessed via the URL. 

### Training
The models are trained on the dataset provided in the [research](https://www.sciencedirect.com/science/article/pii/S0168169999000460) by Jock A. Blackard and Denis J. Dean in the 1990s. The data is provided primarily by U.S. Geological Survey(USGS) and U.S. Forest Survey (USFS). The models are trained on AMD Radeon Pro 5300M CPUs. 

## Australian Bushfire Analysis
### Purpose
The 2019-2020 Australian Bushfire season had unprecedented fire conditions, and one of the dire consequences is the air pollution it resulted in. In this project, a brief visualization of the fire conditions is demonstrated, followed by the implementation of various statistical/ML classification models in attempt to predict the level of PM2.5 and PM10 in Adelaide, Brisbane, and Sydney given the satellite data of the location and severity of the fire cases in Australia at a given time. Moreover, the relationship between the confidence level of a fire instance (provided by the satellite data) and other satellite-observed instance attributes is explored.

### Model
After attempting various regression models including SVR (with linear or RBF kernel), linear regression, and ANN (with Rmsprop/Adam optimizer), it is proven that the dataset has insufficient/biased data for regression modelling. Thus, the regression problem has turned into a single-label classification problem -- given the fire instances in Australia, predict the PM10 *level* (from 4 classes) in one of the three cities on a given day. A variety of ML statistical models are attempted such as KNN, Logistic Regression (LR), Decision Tree, Random Forest, and SVC, and it is concluded that the Decision Tree model is best for Sydney and Adelaide, and the KNN model is best for Brisbane, despite of the fact that the F1-scores of those models are poor and range between 0.6-0.8 due to the limitation of the dataset. In exploration of the relationship between the confidence level of a fire instance and other instance attributes, after attempting KNN, LR, Decision Tree, and Random Forest, it is concluded that the Random Forest model is best-performing and has a 0.96 F1-score.

### Training
The models are trained on the satellite fire instances data, and the air quality data of Adelaide, Brisbane and Sydney. The fire instances dataset is [NASA's VIIRS I-Band 375m Active Fire Data](https://earthdata.nasa.gov/earth-observation-data/near-real-time/firms/viirs-i-band-active-fire-data); The datasets of the cities' air quality are downloaded from [South Australian Government Data Directory](https://data.sa.gov.au/data/dataset/adelaide-cbd-air-quality-monitoring-station-particle), [Queensland Government Environment, Land and Water](https://apps.des.qld.gov.au/air-quality/download/), and [NSW Department of Planning, Industry and Environment](https://www.dpie.nsw.gov.au/air-quality/search-for-and-download-air-quality-data). The models are trained on AMD Radeon Pro 5300M CPUs.


## Fathead Minnow Toxicity Analysis
### Purpose
Fathead minnow (*Pimephales promelas*) is a species of freshwater fish that is commonly used to analyze various chemicals' and toxins' effects on aquatic organisms. In this project, a variety of statistical algorithms and an ANN is attempted to predict the LC50 toxicity level in Fathead minnow given the concentration level of six derived molecular descriptors.

### Model
Regression models including SVR and linear regression, and an Artificial Neural Network (ANN) are implemented using Scikit-learn. It is concluded that the SVR model with a radial basis function kernel is the best-performing model with an RMSE of 0.805.

### Training
The model is trained on a dataset of 908 instances of fish toxicity data collected in the [research by M.Cassotti et al](https://www.tandfonline.com/doi/full/10.1080/1062936X.2015.1018938?scroll=top&needAccess=true&). The models are trained on AMD Radeon Pro 5300M CPUs. 
