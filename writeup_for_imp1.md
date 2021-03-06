# **Traffic Sign Recognition**

This writeup is for `Traffic_Sign_Classifier.ipynb`. Implement with tensorflow

Here is a link to my [project code](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

## Overview

In this project, I use what I have learned about deep neural networks and convolutional neural networks to classify traffic signs. I train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I try out your model on images of German traffic signs that I found on the web.

The goals & steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing distribution of the training set.

![](http://p37mg8cnp.bkt.clouddn.com/201805172333_769.png)

Number of examples per label is different(some have more than others) .



![](http://p37mg8cnp.bkt.clouddn.com/201805172333_368.png)

We can figure out that three dataset are almost identically distribution.

Here are some image from training set

![](http://p37mg8cnp.bkt.clouddn.com/201805211350_708.png)



### Design and Test a Model Architecture

#### 1. Preprocessing

At first I try to convert the images to grayscale, but it doesn't help a lot. So the code is commented out.

The only step I used to preprocess the data is normalization. I normalized the image data because normalization can make the training faster and reduce the chance of getting stuck in local optima. 

```python
def grayscale(X):
    r, g, b = X[:,:,:,0], X[:,:,:,1], X[:,:,:,2]
    X_grayscale = 0.2989 * r + 0.5870 * g + 0.1140 * b
    X_grayscale = X_grayscale.reshape([-1,32,32,1])
    return X_grayscale

def normalization(X):
    # normalize
    X_normalized = X / 255.
    return X_normalized

def preprocess(X):
    # greyscale
#     X = grayscale(X)
    
    # normalize
    X_normalized = normalization(X)
    return X_normalized
```

I will try augment the training data later.

#### 2. Model Architecture

I use Lenet-5 to classify the traffic signs. The model consisted of the following layers:

| Layer         		|     Description	        					|Input|Output|
|:---------------------:|:---------------------------------------------:|:---------------------------------------------:|:---------------------------------------------:|
| Convolution 5x5    | 1x1 stride, valid padding, RELU activation |32x32x3|28x28x6|
| Max pooling	      	| 2x2 ksize, 2x2 stride |28x28x6|14x14x6|
| Convolution 5x5	  | 1x1 stride, valid padding, RELU activation |14x14x6|10x10x16|
| Max pooling	| 2x2 ksize, 2x2 stride |10x10x16|5x5x16|
| Flatten | convert 3 dimensions to 1 dimension |5x5x16|400|
| Fully connected | FC with l2 regularization |400|120|
| Fully connected | FC with l2 regularization |120|84|
| Fully connected | FC with l2 regularization |84|43|
| Softmax |  |||

![](http://p37mg8cnp.bkt.clouddn.com/201805211422_920.png)

Generated by tensorboard

#### 3. Model Training

To train the model, I used an AdamOptimizer with default parameters:

* beta1 = 0.9
* beta2 = 0.999
* epsilon = 1e-08

learning rate with exponential decay

- initial learning rate = 0.01
- decay steps = 100
- decay rate = 0.98 

Other hyper-parameters

* EPOCHS = 60
* BATCH_SIZE = 256
* SIGMA  = 0.1
* weight_decay_rate = 0.005

My final model results were:
* training set accuracy of 0.994
* validation set accuracy of 0.970
* test set accuracy of 0.932

#### 4. Solution Approach

At first I use Lenet-5 with default hyper-parameters shown in the Lenet lab, because Lenet-5 is good at solving image classification problem. And when I try it on traffic signs dataset, it can learn something. I got a validation accuracy of about 0.89.

There were also some problems with the initial architecture:

* learn too slow
* sometimes accuracy fluctuating  

I tryed to tune the initial learning rate and apply learning rate decay to learn faster, but if it is too big, it may cause unable convergence.

I tryed to tune the number of epochs to avoid under fitting.

I tryed to Increase batch size to reduce fluctuating.

I tryed to use l2 regularization and tune the weight_decay_rate to avoid over fitting.

Hyper-parameters tuned:

* Initial learning rate: 0.001, 0.005, 0.02, before settling on 0.01
* BATCH_SIZE: 64, 128, 512, before settling on 256
* EPOCHS:10, 20, 40, 80, before settling on 60
* weight_decay_rate: 0.001,0.002, 0.01, before settling on 0.005

Convolutional layers work well with this problem because each layer detects features automatically which are very useful to classify the traffic sign.  L2 regularization can prevent overfitting by decrease the weights, because of  none of the neurons or the features are relied upon excessively.

### Test a Model on New Images

#### 1. Acquiring New Images

Here are fifteen German traffic signs that I found on the web:

![](http://p37mg8cnp.bkt.clouddn.com/201805212013_242.png)

The signs "Children crossing", " Road work ", "Right-of-way at the next intersection" and "Slippery road" may not be easy to detect, because they are similar and The sign "Keep right" is wearout. Others should be easy to detect.

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children crossing | Bicycles crossing |
| Road work | Beware of ice/snow |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Slippery road	| Slippery road	|
| Turn left ahead	| Turn left ahead |
| Beware of ice/snow	| Beware of ice/snow |
| Priority road	| Priority road |
| Speed limit (80km/h)	| Speed limit (60km/h) |
| End of no passing	| End of all speed and passing limits |
| Yield	| Yield |
| Stop	| Stop |
| Keep right	| Keep right |
| Ahead only	| Ahead only |
| Keep left	| Keep left |
| No entry	| No entry |

![](http://p37mg8cnp.bkt.clouddn.com/201805221239_408.png)

The 2nd, 6th, 8th, 9th images are misclassified.

2nd, 6th, 8th images might be difficult to classify because the test image has some similarities to the prediction image.

The model was able to correctly guess 11 of the 15 traffic signs, which gives an accuracy of 73.3%. The model don't  perform well on the new images.

##### Precision, Recall and F1 score of model on the test set.

| ClassId | SignName                                          | num of training | F1 score | Precision |  Recall  |
| ------- | ------------------------------------------------- | :-------------: | :------: | :-------: | :------: |
| 0       | Speed limit (20km/h)                              |       180       | 0.833333 | 0.937500  | 0.750000 |
| 1       | Speed limit (30km/h)                              |      1980       | 0.907552 | 0.854167  | 0.968056 |
| 2       | Speed limit (50km/h)                              |      2010       | 0.938722 | 0.918367  | 0.960000 |
| 3       | Speed limit (60km/h)                              |      1260       | 0.926554 | 0.942529  | 0.911111 |
| 4       | Speed limit (70km/h)                              |      1770       | 0.926094 | 0.921922  | 0.930303 |
| 5       | Speed limit (80km/h)                              |      1650       | 0.883446 | 0.944043  | 0.830159 |
| 6       | End of speed limit (80km/h)                       |       360       | 0.755187 | 1.000000  | 0.606667 |
| 7       | Speed limit (100km/h)                             |      1290       | 0.908671 | 0.946988  | 0.873333 |
| 8       | Speed limit (120km/h)                             |      1260       | 0.914647 | 0.869739  | 0.964444 |
| 9       | No passing                                        |      1320       | 0.977505 | 0.959839  | 0.995833 |
| 10      | No passing for vehicles over 3.5 metric tons      |      1800       | 0.984663 | 0.996894  | 0.972727 |
| 11      | Right-of-way at the next intersection             |      1170       | 0.943089 | 0.920635  | 0.966667 |
| 12      | Priority road                                     |      1890       | 0.934307 | 0.941176  | 0.927536 |
| 13      | Yield                                             |      1920       | 0.983950 | 0.988780  | 0.979167 |
| 14      | Stop                                              |       690       | 0.985130 | 0.988806  | 0.981481 |
| 15      | No vehicles                                       |       540       | 0.934537 | 0.888412  | 0.985714 |
| 16      | Vehicles over 3.5 metric tons prohibited          |       360       | 0.976744 | 0.973510  | 0.980000 |
| 17      | No entry                                          |       990       | 0.967096 | 0.997050  | 0.938889 |
| 18      | General caution                                   |      1080       | 0.845860 | 0.840506  | 0.851282 |
| 19      | Dangerous curve to the left                       |       180       | 0.839161 | 0.722892  | 1.000000 |
| 20      | Dangerous curve to the right                      |       300       | 0.736842 | 0.608696  | 0.933333 |
| 21      | Double curve                                      |       270       | 0.607595 | 0.705882  | 0.533333 |
| 22      | Bumpy road                                        |       330       | 0.867925 | 1.000000  | 0.766667 |
| 23      | Slippery road                                     |       450       | 0.884211 | 0.933333  | 0.840000 |
| 24      | Road narrows on the right                         |       240       | 0.573248 | 0.671642  | 0.500000 |
| 25      | Road work                                         |      1350       | 0.906414 | 0.915074  | 0.897917 |
| 26      | Traffic signals                                   |       540       | 0.827778 | 0.827778  | 0.827778 |
| 27      | Pedestrians                                       |       210       | 0.550459 | 0.612245  | 0.500000 |
| 28      | Children crossing                                 |       480       | 0.932432 | 0.945205  | 0.920000 |
| 29      | Bicycles crossing                                 |       240       | 0.890000 | 0.809091  | 0.988889 |
| 30      | Beware of ice/snow                                |       390       | 0.739300 | 0.887850  | 0.633333 |
| 31      | Wild animals crossing                             |       690       | 0.968921 | 0.956679  | 0.981481 |
| 32      | End of all speed and passing limits               |       210       | 0.736196 | 0.582524  | 1.000000 |
| 33      | Turn right ahead                                  |       599       | 0.971831 | 0.958333  | 0.985714 |
| 34      | Turn left ahead                                   |       360       | 0.939130 | 0.981818  | 0.900000 |
| 35      | Ahead only                                        |      1080       | 0.964646 | 0.950249  | 0.979487 |
| 36      | Go straight or right                              |       330       | 0.987448 | 0.991597  | 0.983333 |
| 37      | Go straight or left                               |       180       | 0.887218 | 0.808219  | 0.983333 |
| 38      | Keep right                                        |      1860       | 0.976070 | 0.976778  | 0.975362 |
| 39      | Keep left                                         |       270       | 0.878788 | 0.805556  | 0.966667 |
| 40      | Roundabout mandatory                              |       300       | 0.839080 | 0.869048  | 0.811111 |
| 41      | End of no passing                                 |       210       | 0.713178 | 0.666667  | 0.766667 |
| 42      | End of no passing by vehicles over 3.5 metric ... |       210       | 0.831169 | 1.000000  | 0.711111 |

Through the form we can see that the classes which are misclassified also have bad precision, recall and f1 score on the test set.

The classes which have more examples in the training set tends to get a better precision, recall and f1 score on the test set.

I think the partly reason for misclassifying is the distribution of the images I found on web is slightly different with the training images.

So I think data augment may help. I will try it later.

#### 3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the 33th~37th cell of the jupyter notebook.

I will not show all the image below. Only choose some specific images to represent.

For the first image, the model is very confident with its prediction that this is a "Right-of-way at the next intersection" sign (probability of ~1.0), and the image does contain a "Right-of-way at the next intersection" sign. The top five soft max probabilities were show below.

![](http://p37mg8cnp.bkt.clouddn.com/201805221221_897.png)



For the second image , the model's prediction is wrong. And the model is not confident with its predictions. The correct solution is on the 2nd place with 0.309 probability. 

![](http://p37mg8cnp.bkt.clouddn.com/201805221222_196.png)

For the third image , the model's prediction is also wrong.The correct solution is on the 4th place with 0.009 probability.

![](http://p37mg8cnp.bkt.clouddn.com/201805221224_217.png)

For the fourth image ,the model is quite confident with its prediction but that is also wrong.The correct solution is on the 4th place with less than 0.001 probability. That's too bad. 

![](http://p37mg8cnp.bkt.clouddn.com/201805221230_498.png)

By visualize the top 5 classes's images. We can figure out that these classes's image is all contain a red triangle and some other pattern in the middle. The neural network may not have learned the features well about the patterns in the middle. But the patterns in the middle are important for classification. 

### Visualizing the Neural Network

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

##### Same class from different distribution

![](http://p37mg8cnp.bkt.clouddn.com/201805221236_244.png)



![](http://p37mg8cnp.bkt.clouddn.com/201805221236_255.png)

##### Same images with a trained network and a completely untrained one

![](http://p37mg8cnp.bkt.clouddn.com/201805221237_862.png)

Convolutional layers  detects features automatically which might be useful to classify.