import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
import pydicom as pdcm
from skimage import exposure
from anot_core import annotation as anot
import utils.inbreast_utils as inutils


# Dataset:
# --------
# The overwritten torch class to dynamically retrive
# x's and y's from the given dataset
# 
# -> d: numpy array containing the imgs and the 
#       corresponding masks
#       Shape: (num_of_images, img_width, img_height,
#               1 img-channel + 2 mask-channels)
# <- x, y: retrieve image-mask pair on-demand when 
#          calling __getitem__ function

class Dataset(torch.utils.data.Dataset):
  def __init__(self, d):
    self.dtst = d

  def __len__(self):
    return len(self.dtst)

  def __getitem__(self, index):
    obj = self.dtst[index, :, :, :]
    x = torch.from_numpy(obj[:, :, 0])
    temp = np.array(obj[:, :, 1:3])
    
    y = temp
    y = np.rollaxis(y, 2, 1)
    y = np.rollaxis(y, 1, 0)

    return x, y

# Dataset:
# --------
# The overwritten torch class to dynamically retrive
# x's and y's from the given dataset
# 
# -> d: numpy array containing the imgs and the 
#       corresponding masks
#       Shape: (num_of_images, img_width, img_height,
#               1 img-channel + 2 mask-channels)
# <- x, y: retrieve image-mask pair on-demand when 
#          calling __getitem__ function

class Dataset_d(torch.utils.data.Dataset):
  def __init__(self, path, ssize, offset):
    self.dtst_loc = path
    self.ssize = ssize
    self.offset = offset

  def __len__(self):
    #files = os.listdir(self.dtst_loc)
    return self.ssize

  def __getitem__(self, index):
    filename = str(index + self.offset)
    obj = np.load(self.dtst_loc + str(index) + '.npy')
    obj = (obj - 0.0) / 4095.0
    obj = exposure.equalize_hist(obj)
    x = torch.from_numpy(obj[:, :, 0])
    temp = np.array(obj[:, :, 1:3])
    
    y = temp
    y = np.rollaxis(y, 2, 1)
    y = np.rollaxis(y, 1, 0)

    return x, y


# Create_dataset:
# ---------------
# Builds a new dataset by using augmentations of the 
# original images
# 
# -> args: configuration parameters
# -> paths: dictionary containing paths to original
#           dataset, csv, etc
# -> d_dict: dictionary of the original dataset
# <- dataset: numpy array with the augmented dataset

def create_dataset(args, paths, d_dict):

  idx    = 0
  l      = len(d_dict)
  c_size = (args['dsize'], args['dsize'])

  dataset = np.zeros((l*4, args['dsize'], args['dsize'], 4))
  d_indx  = 0
  for i in tqdm(range(l)):
    cat = inutils.extract_img_cat(d_dict[i])
    dcm = pdcm.dcmread(paths['dcm'] + d_dict[idx]["filename"] + '.dcm')
    img = dcm.pixel_array
    crpd, brdrs = inutils.crop_image(img)
    rszd_img = cv2.resize(crpd, c_size, interpolation=cv2.INTER_LINEAR)
    mask_obj = anot.Annotation(paths['xml'], d_dict[idx]["filename"], img.shape)
    mask = mask_obj.mask[brdrs[2]:brdrs[3], brdrs[0]:brdrs[1], :]
    rszd_mask = cv2.resize(mask, c_size, interpolation=cv2.INTER_LINEAR)
    img_augs = inutils.img_augmentation(rszd_img, rszd_mask)

    for obj in img_augs:
      dataset[d_indx, :, :, 0] = obj['img']
      dataset[d_indx, :, :, 1:4] = obj['mask']
      d_indx += 1

    idx += 1

  return dataset


# Create_data_loader:
# -------------------
# Creates the dynamic loaders which are essential for
# the training and the validation procedures
# 
# -> dataset: numpy array with the augmented dataset
# -> chunk_size: percentage of training size
# <- train_set_ldr: training set loader
# <- test_set_ldr: testing set loader
def create_data_loaders(dataset, chunk_size = 0.8):
  dataset = np.random.permutation(dataset)

  train_size    = int(chunk_size * len(dataset))
  train         = dataset[:train_size, :, :, :]
  test          = dataset[train_size:, :, :, :]
  train_set     = Dataset(train)
  params        = {'batch_size': 50, 'shuffle': True}
  train_set_ldr = torch.utils.data.DataLoader(train_set, **params)
  test_set      = Dataset(test)
  params        = {'batch_size': 50, 'shuffle': False}
  test_set_ldr  = torch.utils.data.DataLoader(test_set, **params)

  return train_set_ldr, test_set_ldr

# Create_data_loader:
# -------------------
# Creates the dynamic loaders which are essential for
# the training and the validation procedures
# 
# -> dataset: numpy array with the augmented dataset
# -> chunk_size: percentage of training size
# <- train_set_ldr: training set loader
# <- test_set_ldr: testing set loader
def create_data_loaders_d(path, dtst_size, chunk_size = 0.8):

  train_size    = int(chunk_size * dtst_size)
  train_set     = Dataset_d(path, train_size, 0)
  params        = {'batch_size': 10, 'shuffle': True}
  train_set_ldr = torch.utils.data.DataLoader(train_set, **params)
  test_set      = Dataset_d(path, dtst_size - train_size, train_size + 1)
  params        = {'batch_size': 10, 'shuffle': False}
  test_set_ldr  = torch.utils.data.DataLoader(test_set, **params)

  return train_set_ldr, test_set_ldr


def abn_patch(patch):
  p_area = patch.shape[0] ** 2
  abn_percent = 0.02 * p_area
  has_tumor = np.sum(patch[:, :, 1]) > abn_percent
  has_calc = False #np.sum(patch[:, :, 2]) > 5

  return has_tumor or has_calc

def h_patch(patch):
  
  healthy = np.sum(patch[:, :, 1]) == 0
  pixels = patch.shape[0] ** 2
  b_cover = 0.9 * pixels
  valid = np.count_nonzero(patch[:, :, 0]) >= b_cover 

  return healthy and valid


def valid_patch(img_shape, h, w, p_size):
  y_overflow = w + p_size > img_shape[1] - 1
  x_overflow = h + p_size > img_shape[0] - 1

  return not (x_overflow or y_overflow)


def extract_patches(img, args):
  p_size = args['p_size']
  patches = []
  height = img.shape[0]
  width = img.shape[1]
  for h in range(0, height, args['p_step']):
    for w in range(0, width, args['p_step']):
      if len(patches) > args['ppi']:
        continue
      vld = valid_patch(img.shape, h, w, p_size)
      if not vld:
        continue

      c_patch = img[h:h+p_size, w:w+p_size, :]
      abn = abn_patch(c_patch)
      if not abn:
        continue
      
      patches.append(c_patch)
  
  h_idx = 0
  for h in range(0, height, args['p_step']):
    for w in range(0, width, args['p_step']):
      if h_idx > args['h_ppi']:
        continue
      vld = valid_patch(img.shape, h, w, p_size)
      if not vld:
        continue

      c_patch = img[h:h+p_size, w:w+p_size, :]
      healthy = h_patch(c_patch)
      if not healthy:
        continue
      
      h_idx += 1
      patches.append(c_patch)
  
  return patches


def create_dataset(args, paths, d_list):

  idx    = 0
  l      = len(d_list)
  tmp_list = []
  imgs_list = []
  
  d_indx  = 0
  p_idx = 0
  dataset_size = 0
  for i in tqdm(range(l)):
    img_name = d_list[idx]
    dcm = pdcm.dcmread(paths['dcm'] + img_name + '.dcm')
    img = dcm.pixel_array
    crpd, brdrs = inutils.crop_image(img)
    
    mask_obj = anot.Annotation(paths['xml'], img_name, img.shape)
    mask = mask_obj.mask[brdrs[2]:brdrs[3], brdrs[0]:brdrs[1], :]
    img_obj = np.zeros((crpd.shape[0], crpd.shape[1], 2))
    img_obj[:, :, 0] = crpd
    img_obj[:, :, 1] = mask[:, :, 0]
    #img_obj[:, :, 2] = mask[:, :, 1]
    tmp_list = extract_patches(img_obj, args)
    
    d_indx += len(tmp_list)
    tmp_patches = np.zeros((len(tmp_list), args['p_size'], args['p_size'], 2))
    # print(len(tmp_list))

    for j in range(len(tmp_list)):
      # tmp_patches[j, :, :, :] = tmp_list[j]
      np.save(paths['dtst']+str(p_idx), tmp_list[j])
      file_name = paths['dtst']+str(p_idx)+'.npy'
      file_stats = os.stat(file_name)
      mb = file_stats.st_size / (1024 * 1024)
      dataset_size += mb
      p_idx += 1
    
    if dataset_size > 10000:
      break
    
    idx += 1
  return dataset_size
