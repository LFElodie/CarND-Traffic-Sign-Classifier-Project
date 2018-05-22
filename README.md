## Project: Build a Traffic Sign Recognition Program

[![Udacity - Self-Driving Car NanoDegree](https://camo.githubusercontent.com/5b9aa393f43d7bb9cc6277140465f5625f2dae7c/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f756461636974792d7364632f6769746875622f736869656c642d6361726e642e737667)](http://www.udacity.com/drive)

## Overview

In this project, I use what I have learned about deep neural networks and convolutional neural networks to classify traffic signs. I train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I try out your model on images of German traffic signs that I found on the web.

## The Project

The goals / steps of this project are the following:

- Load the data set
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

### Dependencies

This lab requires:

- [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Implement

I have two implement and two writeup.

The first one implement with tensorflow. 

*  [Traffic_Sign_Classifier.ipynb](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)
* Lenet-5 with l2 regularization without data augment.
* For details see  [writeup_for_imp1.md](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_for_imp1.md)

The second one implement with keras. 

* [Traffic_Sign_Classifier_with_keras.ipynb](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_with_keras.ipynb)
* Lenet-5 with dropout with data augment
* For details see  [writeup_for_imp2.md](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_for_imp2.md)