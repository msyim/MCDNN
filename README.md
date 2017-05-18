# MCDNN

A pytorch implementation of MCDNN introduced in the paper : 

http://people.idsia.ch/~ciresan/data/cvpr2012.pdf

### Model 

MCDNN(multi-column deep neural network) architecture is described in the picture below:
![](https://lh3.googleusercontent.com/-oapqUuBsBaE/Vhd60PngQQI/AAAAAAAAABI/0X5eRPiX8rI/s320/MCDNN.PNG)

### Data augmentation
* Zoomed-in and out images : 10 different dimensions (including the original) have been used to train the model
* (Below 4 augmentations are done using the ImageDataGenerator module in Keras)
* ZCA whitening
* Feature standardization
* Random Rotation up to 90 degrees
* Random Shift

### thoughts
Data augmentation (except for zooming-in and out) doesn't seem to be helping (in fact, the accuracy is far lower than the one not using the 4 ImageDataGenerator methods).  Need to find out the optimal frequency of introducing augmented data to the original set of images.
