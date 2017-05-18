# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

# Numpy
import numpy as np

# Scipy to resize images
from scipy.misc import imresize

# Keras for easy data augmentation
from keras.preprocessing.image import ImageDataGenerator


class mcdnn(nn.Module):
  def __init__(self):
    super(mcdnn,self).__init__()

    # Using the original mnist data
    self.n1c1 = nn.Conv2d(1,20, kernel_size=4,padding=1)
    self.n1bn1 = nn.BatchNorm2d(20)
    self.n1c2 = nn.Conv2d(20,40, kernel_size=5,padding=1)
    self.n1bn2 = nn.BatchNorm2d(40)
    self.n1fc1 = nn.Linear(360, 150)
    self.n1fc2 = nn.Linear(150,10)

    # 36 * 36
    self.n0c1 = nn.Conv2d(1,20, kernel_size=4,padding=1)
    self.n0bn1 = nn.BatchNorm2d(20)
    self.n0c2 = nn.Conv2d(20,40, kernel_size=5,padding=1)
    self.n0bn2 = nn.BatchNorm2d(40)
    self.n0fc1 = nn.Linear(1000, 200)
    self.n0fc2 = nn.Linear(200,10)

    # 40 * 40
    self.n9c1 = nn.Conv2d(1,20, kernel_size=4,padding=1)
    self.n9bn1 = nn.BatchNorm2d(20)
    self.n9c2 = nn.Conv2d(20,40, kernel_size=5,padding=1)
    self.n9bn2 = nn.BatchNorm2d(40)
    self.n9fc1 = nn.Linear(1000, 250)
    self.n9fc2 = nn.Linear(250,10)

    # 50 * 50
    self.n8c1 = nn.Conv2d(1,20, kernel_size=4,padding=1)
    self.n8bn1 = nn.BatchNorm2d(20)
    self.n8c2 = nn.Conv2d(20,40, kernel_size=5,padding=1)
    self.n8bn2 = nn.BatchNorm2d(40)
    self.n8fc1 = nn.Linear(1960, 500)
    self.n8fc2 = nn.Linear(500, 150)
    self.n8fc3 = nn.Linear(150,10)

    # 20 * 20
    self.n2c1 = nn.Conv2d(1,20, kernel_size=4,padding=1)
    self.n2bn1 = nn.BatchNorm2d(20)
    self.n2c2 = nn.Conv2d(20,40, kernel_size=5,padding=1)
    self.n2bn2 = nn.BatchNorm2d(40)
    self.n2fc1 = nn.Linear(360, 80)
    self.n2fc2 = nn.Linear(80,10)

    # 18 * 18
    self.n3c1 = nn.Conv2d(1,20, kernel_size=3,padding=1)
    self.n3bn1 = nn.BatchNorm2d(20)
    self.n3c2 = nn.Conv2d(20,40, kernel_size=4,padding=1)
    self.n3bn2 = nn.BatchNorm2d(40)
    self.n3fc1 = nn.Linear(640, 150)
    self.n3fc2 = nn.Linear(150,10)

    # 16 * 16
    self.n4c1 = nn.Conv2d(1,20, kernel_size=3,padding=1)
    self.n4bn1 = nn.BatchNorm2d(20)
    self.n4c2 = nn.Conv2d(20,40, kernel_size=4,padding=1)
    self.n4bn2 = nn.BatchNorm2d(40)
    self.n4fc1 = nn.Linear(360, 150)
    self.n4fc2 = nn.Linear(150,10)

    # 14 * 14
    self.n5c1 = nn.Conv2d(1,20, kernel_size=3,padding=1)
    self.n5bn1 = nn.BatchNorm2d(20)
    self.n5c2 = nn.Conv2d(20,40, kernel_size=4,padding=1)
    self.n5bn2 = nn.BatchNorm2d(40)
    self.n5fc1 = nn.Linear(360, 150)
    self.n5fc2 = nn.Linear(150,10)

    # 12 * 12
    self.n6c1 = nn.Conv2d(1,20, kernel_size=2,padding=1)
    self.n6bn1 = nn.BatchNorm2d(20)
    self.n6c2 = nn.Conv2d(20,40, kernel_size=3,padding=1)
    self.n6bn2 = nn.BatchNorm2d(40)
    self.n6fc1 = nn.Linear(360, 80)
    self.n6fc2 = nn.Linear(80,10)

    # 10 * 10
    self.n7c1 = nn.Conv2d(1,20, kernel_size=2,padding=1)
    self.n7bn1 = nn.BatchNorm2d(20)
    self.n7c2 = nn.Conv2d(20,40, kernel_size=3,padding=1)
    self.n7bn2 = nn.BatchNorm2d(40)
    self.n7fc1 = nn.Linear(160, 80)
    self.n7fc2 = nn.Linear(80,10)

  def forward(self,x):
    # Using the original mnist data
    #n1_out = F.relu(F.max_pool2d(self.n1c1(x),2))
    #n1_out = F.relu(F.max_pool2d(self.n1c2(n1_out),3))
    n1_out = F.relu(F.max_pool2d(self.n1bn1(self.n1c1(x)),2))
    n1_out = F.relu(F.max_pool2d(self.n1bn2(self.n1c2(n1_out)),3))
    n1_out = n1_out.view(-1, 360)
    n1_out = F.relu(self.n1fc1(n1_out))
    n1_out = self.n1fc2(n1_out)

    # Scale to 20 * 20
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(20,20)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,20,20))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
   
    #n2_out = F.relu(F.max_pool2d(self.n2c1(in_resized),2))
    #n2_out = F.relu(F.max_pool2d(self.n2c2(n2_out),2))
    n2_out = F.relu(F.max_pool2d(self.n2bn1(self.n2c1(in_resized)),2))
    n2_out = F.relu(F.max_pool2d(self.n2bn2(self.n2c2(n2_out)),2))
    n2_out = n2_out.view(-1, 360)
    n2_out = F.relu(self.n2fc1(n2_out))
    n2_out = self.n2fc2(n2_out)

    # Scale to 18 * 18
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(18,18)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,18,18))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n3_out = F.relu(F.max_pool2d(self.n3c1(in_resized),2))
    #n3_out = F.relu(F.max_pool2d(self.n3c2(n3_out),2))
    n3_out = F.relu(F.max_pool2d(self.n3bn1(self.n3c1(in_resized)),2))
    n3_out = F.relu(F.max_pool2d(self.n3bn2(self.n3c2(n3_out)),2))
    n3_out = n3_out.view(-1, 640)
    n3_out = F.relu(self.n3fc1(n3_out))
    n3_out = self.n3fc2(n3_out)

    # Scale to 16 * 16
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(16,16)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,16,16))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n4_out = F.relu(F.max_pool2d(self.n4c1(in_resized),2))
    #n4_out = F.relu(F.max_pool2d(self.n4c2(n4_out),2))
    n4_out = F.relu(F.max_pool2d(self.n4bn1(self.n4c1(in_resized)),2))
    n4_out = F.relu(F.max_pool2d(self.n4bn2(self.n4c2(n4_out)),2))
    n4_out = n4_out.view(-1, 360)
    n4_out = F.relu(self.n4fc1(n4_out))
    n4_out = self.n4fc2(n4_out)

    # Scale to 14 * 14
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(14,14)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,14,14))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n5_out = F.relu(F.max_pool2d(self.n5c1(in_resized),2))
    #n5_out = F.relu(F.max_pool2d(self.n5c2(n5_out),2))
    n5_out = F.relu(F.max_pool2d(self.n5bn1(self.n5c1(in_resized)),2))
    n5_out = F.relu(F.max_pool2d(self.n5bn2(self.n5c2(n5_out)),2))
    n5_out = n5_out.view(-1, 360)
    n5_out = F.relu(self.n5fc1(n5_out))
    n5_out = self.n5fc2(n5_out)

    # Scale to 12 * 12
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(12,12)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,12,12))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n6_out = F.relu(F.max_pool2d(self.n6c1(in_resized),2))
    #n6_out = F.relu(F.max_pool2d(self.n6c2(n6_out),2))
    n6_out = F.relu(F.max_pool2d(self.n6bn1(self.n6c1(in_resized)),2))
    n6_out = F.relu(F.max_pool2d(self.n6bn2(self.n6c2(n6_out)),2))
    n6_out = n6_out.view(-1, 360) 
    n6_out = F.relu(self.n6fc1(n6_out))
    n6_out = self.n6fc2(n6_out)

    # Scale to 10 * 10
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(10,10)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,10,10))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n7_out = F.relu(F.max_pool2d(self.n7c1(in_resized),2))
    #n7_out = F.relu(F.max_pool2d(self.n7c2(n7_out),2))
    n7_out = F.relu(F.max_pool2d(self.n7bn1(self.n7c1(in_resized)),2))
    n7_out = F.relu(F.max_pool2d(self.n7bn2(self.n7c2(n7_out)),2))
    n7_out = n7_out.view(-1, 160)
    n7_out = F.relu(self.n7fc1(n7_out))
    n7_out = self.n7fc2(n7_out)

    # Scale to 50 * 50
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(50,50)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,50,50))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n8_out = F.relu(F.max_pool2d(self.n8c1(in_resized),2))
    #n8_out = F.relu(F.max_pool2d(self.n8c2(n8_out),2))
    n8_out = F.relu(F.max_pool2d(self.n8bn1(self.n8c1(in_resized)),2))
    n8_out = F.relu(F.max_pool2d(self.n8bn2(self.n8c2(n8_out)),3))
    n8_out = n8_out.view(-1, 1960)
    n8_out = F.relu(self.n8fc1(n8_out))
    n8_out = F.relu(self.n8fc2(n8_out))
    n8_out = self.n8fc3(n8_out)

    # Scale to 40 * 40
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(40,40)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,40,40))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n9_out = F.relu(F.max_pool2d(self.n9c1(in_resized),2))
    #n9_out = F.relu(F.max_pool2d(self.n9c2(n9_out),3))
    n9_out = F.relu(F.max_pool2d(self.n9bn1(self.n9c1(in_resized)),2))
    n9_out = F.relu(F.max_pool2d(self.n9bn2(self.n9c2(n9_out)),3))
    n9_out = n9_out.view(-1, 1000)
    n9_out = F.relu(self.n9fc1(n9_out))
    n9_out = self.n9fc2(n9_out)

    # Scale to 36 * 36
    in_resized = []
    for el in x:
      temp = np.reshape(el.cpu().data.numpy(), (28,28))
      in_resized.append(imresize(temp,(36,36)).astype(float)/255)
    in_resized = np.reshape(np.asarray(in_resized), (100,1,36,36))
    in_resized = Variable(torch.Tensor(in_resized)).cuda()
    
    #n0_out = F.relu(F.max_pool2d(self.n0c1(in_resized),2))
    #n0_out = F.relu(F.max_pool2d(self.n0c2(n0_out),3))
    n0_out = F.relu(F.max_pool2d(self.n0bn1(self.n0c1(in_resized)),2))
    n0_out = F.relu(F.max_pool2d(self.n0bn2(self.n0c2(n0_out)),3))
    n0_out = n0_out.view(-1, 1000)
    n0_out = F.relu(self.n0fc1(n0_out))
    n0_out = self.n0fc2(n0_out)

    net_result = (n1_out + n2_out + n3_out + n4_out + n5_out + n6_out + n7_out + n8_out + n9_out + n0_out)/10
    return F.log_softmax(net_result)
