# Pixel Level Segmentation Based Drivable Road Region Detection and Steering Angle Estimation Method for Autonomous Driving on Unstructured Roads
# Discription of Project 
A PyTorch implementation of the  Road Dataset semantic segmentation using FULLY CONVOLUTIONAL ResNet50 FPN model. The dataset has been taken from CARL-DATASET (https://carl-dataset.github.io/index/).

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
we have extended CARL-Dataset for road detection and segmentation task. As CARL-Dataset has been constructed over video sequences from 100+ cities of Pakistan. Consequently, this dataset contains diversities in terms of road types such as (i) highways (ii) motorways (iii) rural and urban streets (iv) provincial, (v) district, and (iv) hilly and distressed roads. To ensure the generalization of our proposed method, equal subsets of images from video sequences of all types of captured roads have been selected for the training and evaluation of proposed method. It is important to mention that in this research work, roads with missing lane-markings, concrete and distressed roads are categorized as unstructured roads.
