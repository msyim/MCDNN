# Tensorflow : used TF to load mnist data for consistency
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Numpy
import numpy as np

# Torch : core ML functions are implemented with PyTorch
import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

# Keras : for easy data augmentation
from keras.preprocessing.image import ImageDataGenerator

# My implementation of (a modified version of) MultiColumn DNN
# based on the paper "Multi-column Deep Neural Networks for Image Classification
# paper url : http://people.idsia.ch/~ciresan/data/cvpr2012.pdf
from mcdnn import mcdnn

# HyperParameters
EPOCH=800
BATCH_SIZE=100
LEARNING_RATE=0.001

# Read input data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set the model
#mymodel = mininet()
mymodel = mcdnn()
mymodel.cuda()

#Loss and Optimizer
cost = tnn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(mymodel.parameters(), lr=LEARNING_RATE)

#Train the model
for epoch in range(EPOCH):
  avg_cost = 0
  add_iter = 0
  total_batch = mnist.train.num_examples/BATCH_SIZE

  LEARNING_RATE = max(0.00003, (LEARNING_RATE * 0.993))
  print('current learning rate: %f' % LEARNING_RATE)
  for i in range(total_batch):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    batch_x = np.reshape(batch_x, (BATCH_SIZE,1,28,28))
    batch_y = np.argmax(batch_y,axis=1)

    images = Variable(torch.Tensor(batch_x)).cuda()
    labels = Variable(torch.LongTensor(batch_y)).cuda()

    optimizer = torch.optim.Adam(mymodel.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad()
    model_output = mymodel.forward(images)
    loss = cost(model_output, labels)
    avg_cost += loss.data[0]
    loss.backward()
    optimizer.step()

    # Data agumentation in every 4 steps
    if ((i+1) % 4 == 0) :
      # ZCA whitening
      if ((i+1) % 16 == 0) :
        print('ZCA whitening') 
        datagen = ImageDataGenerator(zca_whitening=True)
      # Feature Standardization
      elif ((i+1) % 12 == 0) :
        print('Feature Standardization') 
        datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
      # Random Rotation up to 90 degrees
      elif ((i+1) % 8 == 0) :
        print('Random Rotation') 
        datagen = ImageDataGenerator(rotation_range=90)
      # Random shift
      else :
        print('Random Shift') 
        datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
      
      batch_x = np.reshape(batch_x, (BATCH_SIZE,28,28,1))
      datagen.fit(batch_x)
      for aug_batch_x in datagen.flow(batch_x, batch_size=BATCH_SIZE):
        aug_batch_x = np.reshape(aug_batch_x, (BATCH_SIZE,1,28,28))
        images = Variable(torch.Tensor(aug_batch_x)).cuda()

        optimizer = torch.optim.Adam(mymodel.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad()
        model_output = mymodel.forward(images)
        loss = cost(model_output, labels)
        avg_cost += loss.data[0]
        add_iter += 1
        loss.backward()
        optimizer.step()
        break

    if ((i+1)%1 == 0) :
      print( 'Epoch [%d/%d], Iter[%d/%d] avg Loss. %.4f' %
        (epoch+1, EPOCH, i+1, total_batch, avg_cost/(add_iter + i + 1)))

  test_batch = mnist.test.num_examples/BATCH_SIZE
  accuracy = 0

  for i in range(test_batch):
    test_x, test_y = mnist.test.next_batch(BATCH_SIZE)
    test_x = np.reshape(test_x, (len(test_x),1,28,28))
    test_y = np.argmax(test_y, axis=1)

    test_images = Variable(torch.Tensor(test_x)).cuda()
    test_labels = test_y

    test_output = mymodel(test_images)
    test_output = np.argmax(test_output.cpu().data.numpy(), axis=1)
  #test_output = torch.argmax(test_output, axis=1)
  #print(test_labels.shape)
  #print(test_output.shape)

    accuracy_temp = float(np.sum(test_labels == test_output))
    accuracy += accuracy_temp
    if ((i+1)%10 == 0 ) :
      print("Epoch [%d/%d], TestBatch [%d/%d] batch acc: %f" % 
          (epoch+1, EPOCH, i+1, test_batch, (accuracy_temp/100)))
  
  print("Accuracy: %f" % (accuracy/mnist.test.num_examples))

