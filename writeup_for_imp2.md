# **Traffic Sign Recognition**

This writeup is for `Traffic_Sign_Classifier_with_keras.ipynb`. Implement with keras

Here is a link to my [project code](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier_with_keras.ipynb)

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

At first Step, I decided to use Histogram Equalization. That can normalizes the brightness and increases the contrast of the image. 

Then I normalized the image because normalization can make the training faster and reduce the chance of getting stuck in local optima. 

```python
def equalize(X):
    equ = np.zeros_like(X)
    r, g, b = X[:,:,0], X[:,:,1], X[:,:,2]
    equ[:,:,0] += equalizeHist(r)
    equ[:,:,1] += equalizeHist(g)
    equ[:,:,2] += equalizeHist(b)
    return equ
def normalization(X):
    # normalize
    X_normalized = X / 255.
    return X_normalized
def preprocess(X):
    # normalize
    X_preprocessed = np.zeros_like(X)
    for i in range(len(X)):
        X_preprocessed[i]=equalize(X[i])
    X_normalized = normalization(X_preprocessed)
    return X_normalized
```

Here is an example of a traffic sign image before and after preprocessing. 

![](http://p37mg8cnp.bkt.clouddn.com/201805222230_805.png)

I decided to generate additional data because more data are always helpful in deep learning.

To add more data to the the data set, I used the following techniques :

|  techniques   |                      reason                      |
| :-----------: | :----------------------------------------------: |
|  width shift  |       the sign may not be at exact middle        |
| height shift  |       the sign may not be at exact middle        |
|   rotation    |      the sign may not be at exact vertical       |
|     shear     | the sign may not be exact parallel to the camera |
| channel shift |     lighting conditions may not be constant      |

Here is an example of  augmented image:

![](http://p37mg8cnp.bkt.clouddn.com/201805222239_454.png)

I use `keras.preprocessing.image.ImageDataGenerator ` to

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

#### 3. Model Training

To train the model, I used an AdamOptimizer with default parameters:

* beta1 = 0.9
* beta2 = 0.999
* epsilon = 1e-08

Other hyper-parameters

* EPOCHS = 60
* BATCH_SIZE = 256
* SIGMA  = 0.1
* keep_prob = 0.4

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.962
* test set accuracy of 0.939

#### 4. Solution Approach

At first I use Lenet-5 with default hyper-parameters shown in the Lenet lab, because Lenet-5 is good at solving image classification problem. And when I try it on traffic signs dataset, it can learn something. I got a validation accuracy of about 0.89.

There were also some problems with the initial architecture:

* learn too slow
* sometimes accuracy fluctuating  

I tryed to tune the initial learning rate to learn faster, but if it is too big, it may cause unable convergence.

I tryed to tune the number of epochs to avoid under fitting.

I tryed to Increase batch size to reduce fluctuating.

I tryed to use dropout to avoid over fitting.

Hyper-parameters tuned:

* learning rate:  0.01, 0.005, 0.002 before settling on 0.001
* BATCH_SIZE: 64, 128, 512, before settling on 256
* EPOCHS:10, 20, 40, 80, before settling on 60
* keep_prob = 0.5 ,before settling on 0.4

Convolutional layers work well with this problem because each layer detects features automatically which are very useful to classify the traffic sign.  Dropout can prevent overfitting by decrease the weights, because of  none of the neurons or the features are relied upon excessively.

### Test a Model on New Images

#### 1. Acquiring New Images

Here are fifteen German traffic signs that I found on the web:

![](http://p37mg8cnp.bkt.clouddn.com/201805212013_242.png)

The signs "Children crossing", " Road work ", "Right-of-way at the next intersection" and "Slippery road" may not be easy to detect, because they are similar and The sign "Keep right" is wearout. Others should be easy to detect.

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Children crossing | Road narrows on the right |
| Road work | Road work |
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Slippery road	| Slippery road	|
| Turn left ahead	| Turn left ahead |
| Beware of ice/snow	| Beware of ice/snow |
| Priority road	| Priority road |
| Speed limit (80km/h)	| Speed limit (80km/h) |
| End of no passing	| End of no passing |
| Yield	| Yield |
| Stop	| Stop |
| Keep right	| Keep right |
| Ahead only	| Ahead only |
| Keep left	| Keep left |
| No entry	| No entry |

![](http://p37mg8cnp.bkt.clouddn.com/201805222301_117.png)

The 1st images are misclassified.

The model was able to correctly guess 14 of the 15 traffic signs, which gives an accuracy of 93.33%. The model work well on the new images.

##### Precision, Recall and F1 score of model on the test set.

| ClassId | SignName                                          | num of training | F1 score | Precision | Recall   |
| :-----: | ------------------------------------------------- | --------------- | -------- | --------- | -------- |
|    0    | Speed limit (20km/h)                              | 180             | 0.941176 | 0.949153  | 0.933333 |
|    1    | Speed limit (30km/h)                              | 1980            | 0.973523 | 0.952191  | 0.995833 |
|    2    | Speed limit (50km/h)                              | 2010            | 0.926764 | 0.925532  | 0.928000 |
|    3    | Speed limit (60km/h)                              | 1260            | 0.852713 | 0.849890  | 0.855556 |
|    4    | Speed limit (70km/h)                              | 1770            | 0.942857 | 0.990000  | 0.900000 |
|    5    | Speed limit (80km/h)                              | 1650            | 0.872232 | 0.941176  | 0.812698 |
|    6    | End of speed limit (80km/h)                       | 360             | 0.926174 | 0.932432  | 0.920000 |
|    7    | Speed limit (100km/h)                             | 1290            | 0.868545 | 0.920398  | 0.822222 |
|    8    | Speed limit (120km/h)                             | 1260            | 0.885645 | 0.978495  | 0.808889 |
|    9    | No passing                                        | 1320            | 0.941538 | 0.927273  | 0.956250 |
|   10    | No passing for vehicles over 3.5 metric tons      | 1800            | 0.957813 | 0.988710  | 0.928788 |
|   11    | Right-of-way at the next intersection             | 1170            | 0.845982 | 0.796218  | 0.902381 |
|   12    | Priority road                                     | 1890            | 0.984772 | 0.985486  | 0.984058 |
|   13    | Yield                                             | 1920            | 0.977444 | 0.962315  | 0.993056 |
|   14    | Stop                                              | 690             | 0.965265 | 0.953069  | 0.977778 |
|   15    | No vehicles                                       | 540             | 0.658228 | 0.492891  | 0.990476 |
|   16    | Vehicles over 3.5 metric tons prohibited          | 360             | 0.986842 | 0.974026  | 1.000000 |
|   17    | No entry                                          | 990             | 0.987654 | 0.975610  | 1.000000 |
|   18    | General caution                                   | 1080            | 0.853186 | 0.927711  | 0.789744 |
|   19    | Dangerous curve to the left                       | 180             | 0.975207 | 0.967213  | 0.983333 |
|   20    | Dangerous curve to the right                      | 300             | 0.823529 | 1.000000  | 0.700000 |
|   21    | Double curve                                      | 270             | 0.652482 | 0.901961  | 0.511111 |
|   22    | Bumpy road                                        | 330             | 0.742857 | 0.650000  | 0.866667 |
|   23    | Slippery road                                     | 450             | 0.900621 | 0.843023  | 0.966667 |
|   24    | Road narrows on the right                         | 240             | 0.456693 | 0.783784  | 0.322222 |
|   25    | Road work                                         | 1350            | 0.932540 | 0.890152  | 0.979167 |
|   26    | Traffic signals                                   | 540             | 0.875332 | 0.837563  | 0.916667 |
|   27    | Pedestrians                                       | 210             | 0.396552 | 0.410714  | 0.383333 |
|   28    | Children crossing                                 | 480             | 0.869565 | 0.952381  | 0.800000 |
|   29    | Bicycles crossing                                 | 240             | 0.681416 | 0.566176  | 0.855556 |
|   30    | Beware of ice/snow                                | 390             | 0.571429 | 0.772727  | 0.453333 |
|   31    | Wild animals crossing                             | 690             | 0.909091 | 0.876289  | 0.944444 |
|   32    | End of all speed and passing limits               | 210             | 0.783333 | 0.783333  | 0.783333 |
|   33    | Turn right ahead                                  | 599             | 0.992874 | 0.990521  | 0.995238 |
|   34    | Turn left ahead                                   | 360             | 0.959677 | 0.929688  | 0.991667 |
|   35    | Ahead only                                        | 1080            | 0.967320 | 0.986667  | 0.948718 |
|   36    | Go straight or right                              | 330             | 0.991597 | 1.000000  | 0.983333 |
|   37    | Go straight or left                               | 180             | 0.983333 | 0.983333  | 0.983333 |
|   38    | Keep right                                        | 1860            | 0.970085 | 0.953782  | 0.986957 |
|   39    | Keep left                                         | 270             | 0.931818 | 0.953488  | 0.911111 |
|   40    | Roundabout mandatory                              | 300             | 0.913295 | 0.951807  | 0.877778 |
|   41    | End of no passing                                 | 210             | 0.870229 | 0.802817  | 0.950000 |
|   42    | End of no passing by vehicles over 3.5 metric ... | 210             | 0.957447 | 0.918367  | 1.000000 |

Through the form we can see that the classes which are misclassified also have bad precision, recall and f1 score on the test set.

The classes which have more examples in the training set tends to get a better precision, recall and f1 score on the test set.

I think the partly reason for misclassifying is the distribution of the images I found on web is slightly different with the training images.

Compared with the implement 1 which is very similar with this model, This model have a better performance. I think the data augment help a lot.

#### 3. Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in the 38th~42th cell of the jupyter notebook.

I will not show all the image below. Only choose some specific images to represent.

For the first image, the model is very confident with its prediction that this is a "Right-of-way at the next intersection" sign (probability of ~1.0), and the image does contain a "Right-of-way at the next intersection" sign. The top five soft max probabilities were show below.

![](http://p37mg8cnp.bkt.clouddn.com/201805222320_421.png)

For the second image , the model's prediction is correct. But the model is not very confident with its predictions. It has only 0.476 degree of confidence.

![](http://p37mg8cnp.bkt.clouddn.com/201805222321_975.png)

For the third image , the model's prediction is wrong.The correct solution is on the 2nd place with 0.382 probability.

![](http://p37mg8cnp.bkt.clouddn.com/201805222324_135.png)

By visualize the top 5 classes's images. We can figure out that these classes's image is all contain a red triangle and some other pattern in the middle. The neural network may not have learned the features well about the patterns in the middle. But the patterns in the middle are important for classification. 

### Visualizing the Neural Network

See [Here](https://github.com/LFElodie/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) 