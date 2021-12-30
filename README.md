# Pixel Level Segmentation Based Drivable Road Region Detection and Steering Angle Estimation Method for Autonomous Driving on Unstructured Roads
# Discription of Project 
A PyTorch implementation of the  Road Dataset semantic segmentation using FULLY CONVOLUTIONAL ResNet50 FPN model. The dataset has been taken from [CARL-DATASET](https://carl-dataset.github.io/index/ "CARL-DATASET").

![4 01351_ov](https://user-images.githubusercontent.com/71174927/147745717-d3065341-ab39-4c1f-8c7a-e3ec09dd443b.jpg)
![4 13693_ov](https://user-images.githubusercontent.com/71174927/147745756-f0d18207-a9f5-4b88-872b-73b80f1d3731.jpg)
![4 21168_ov](https://user-images.githubusercontent.com/71174927/147745788-05eb26a4-f6a4-4761-87ca-c86e16e761ca.jpg)
![4 24633_ov](https://user-images.githubusercontent.com/71174927/147745840-9df41a44-f05c-4ce5-be18-3a3c221327e4.jpg)
# Prerequisites
1. Windows
2. Anaconda Python
3. PyTorch (https://pytorch.org/get-started/locally/)
4. Albumentations (https://pypi.org/project/albumentations/), (pip install albumentations)
5. Tensorboard (https://pypi.org/project/tensorboard/), (pip install tensorboard)
6. TensorboardX (https://pypi.org/project/tensorboardX/), (pip install tensorboardX)
# Dataset 
we have extended CARL-Dataset for road detection and segmentation task. As CARL-Dataset has been constructed over video sequences from 100+ cities of Pakistan. Consequently, this dataset contains diversities in terms of road types such as (i) highways (ii) motorways (iii) rural and urban streets (iv) provincial, (v) district, and (iv) hilly and distressed roads. To ensure the generalization of our proposed method, equal subsets of images from video sequences of all types of captured roads have been selected for the training and evaluation of proposed method. Download the dataset (link).

 https://user-images.githubusercontent.com/71174927/147775570-bc58cbf2-7bcb-4c8b-9c40-09ffe39feb7f.mp4

# Description of folders and scripts
My project includes the following files scripts and folders:
-  CARL_DATASET folder contains the only the road images and corresponding label images.
-  train_seg_maps folder is used to save the prediction of the trained model on rondomly selected image during validation process.
-  runs folder contains the segmentation logs of the testing data.
-  utils folder contains the files such as (i) helper.py and (ii) metrics.py. 
-  config.py is the configation file, where dataset path and name and number of classes mentioned. Check every other parameters in config.py as per your requirement.
-  dataset.py includes the functions regarding dataset.
-  helper.py includes some helper functions.
-  train.py is used to train the model. 
-  road_detection_test.py is used to infer the model on test image.
-  road_detection_angle_estimation_test.py is used to infer the model on the test image and predict the angle. 
-  SteeringAngleUtils.py is used to compute the steering angle againt the given test image. 
-  test_vid.py is used to detect road on given video input. 
-  model.py is used to train the model from scratch.
-  engine.py is used to save the model as a model.pth.
-  metrics.py includes some helper functions regarding evalution of performance matrics.

# How to run the Python scripts
## For Training 
- Train the model for the first time on the road detection dataset CARL-DATASET.
```
python train.py --resume-training no.
```
## For Testing 
### Test the model on the image 
-  Use this python script to apply pixel level segmentation on any image of your choice.
```
python test_road_detection.py --model-path <path to saved checkpoint/weight file> --input <path to vid>.
```
example: python test.py --model-path model.pth --input abc.jpg
### Test the model on the video 
-  Use this python script to apply pixel level segmentation on any videos of your choice.
```
python test_vid.py --input <path to vid> --model-path <path to saved checkpoint/weight file>.
```
example: python test_vid.py --input DSC_0006.mp4 --model-path model.pth


