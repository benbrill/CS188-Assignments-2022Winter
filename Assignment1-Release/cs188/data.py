import os
import torch
import pickle
import numpy as np
import cv2
import math

def _extract_tensors(data, num=None, feature=False):
  """
  Implement the _extract_tensors function.
  
  Extract the data and labels from tinyplaces dataset object and convert them to
  tensors.

  Input:
  - data: tinyplaces of shape (N, 3072), where N is the total number of examples in the dataset.
  - num: Optional. If provided, the number of samples to keep.
  - feature: Optional. If True, extract features from the data instead of using raw pixel values.

  Returns:
  - x: float32 torch tensor of shape (num, 3, 32, 32)
       if feature is True, x is the torch tensor of shape (num, 3, 256). (however, you do not need to implement this since we implement it for you.)
       value should be between 0 and 1 (you should normalize the data)
  - y: int64 torch tensor of shape (num,)
  """
  
  ##############################################################################
  # TODO: Implement this function. You should try to transform numpy arrays to #
  #tensors and resize them. If subsample, only keep a certain number of samples#
  #after this step, x should be a torch tensor of shape (num, 3, 32, 32)       #
  ##############################################################################
  # Replace "pass" statement with your code
  
  #pass
 
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  
  # if feature is true, we extract the color histograms from the images.
  if feature:
    all_features = []
    
    data['data'] = np.vstack(data['data']).reshape(-1, 3, 32, 32)
    
    for i in range(data['data'].shape[0]):
        chans = cv2.split(data['data'][i].transpose(1,2,0))
        features = []
        colors = ('b', 'g', 'r')
        for (chan, color) in zip(chans, colors):
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            elem = np.max(hist)
            features.append(hist/elem)
        all_features.append(features)
    all_features = torch.tensor(all_features)
    x = all_features.squeeze()

  return x, y


def tinyplaces(path='', num_train=None, num_val=None, feature=False, binary=True, use_gpu=False):
  """
  Return the TinyPlaces dataset.
  This function can also subsample the dataset.

  Inputs:
  - num_train: [Optional] How many samples to keep from the training set.
    If not provided, then keep the entire training set.
  - num_val: [Optional] How many samples to keep from the val set.
    If not provided, then keep the entire val set.
  - feature: [Optional] If True, extract features from the images instead of using raw pixel values.
  - binary: [Optional] If false, perform multi-class classification instead of binary classification.

  Returns:
  - x_train: float32 tensor of shape (num_train, 3, 32, 32) / If feature is true, (num_train, 3, 256)
  - y_train: int64 tensor of shape (num_train, )
  - x_val: float32 tensor of shape (num_val, 3, 32, 32) / If feature is true, (num_val, 3, 256)
  - y_val: int64 tensor of shape (num_val, )
  """
  
  if binary:
      train_file = os.path.join(path, "data/tinyplaces-train")
      val_file = os.path.join(path, "data/tinyplaces-val")
  else:
      train_file = os.path.join(path, "data/tinyplaces-train-multiclass")
      val_file = os.path.join(path, "data/tinyplaces-val-multiclass")
      
  with open(train_file, 'rb') as fo1:
    data_train = pickle.load(fo1, encoding='bytes')
  fo1.close()
  
  with open(val_file, 'rb') as fo2:
    data_val = pickle.load(fo2, encoding='bytes')
  fo2.close()
  
  x_train, y_train = _extract_tensors(data_train, num_train, feature)
  x_val, y_val = _extract_tensors(data_val, num_val, feature)
  
  if use_gpu:
      x_train = x_train.cuda()
      y_train = y_train.cuda()
      x_val  = x_val.cuda()
      y_val  = y_val.cuda()
  return x_train, y_train, x_val, y_val
