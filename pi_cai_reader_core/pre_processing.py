from medpy.io import load
import os
import numpy as np
import matplotlib.pyplot as plt

# Build_dict_for_masks:
# ---------------------
# This function creates a dictionary for the
# nifti files locations assigning them to the 
# corresponding patients ids
# 
# -> path: path to the root folder of nifi files
# <- d: dictionary with the nii locations

def build_dict_for_masks(path):
  print("Building dictionary for masks...")
  d = {}
  niis_names = os.listdir(path)

  for nii in niis_names:
    tokens = nii.split('_')
    p_id = tokens[0]
    d[p_id] = nii

  return d


# Fetch_patient_t2w:
# ---------------------
# Retrieves the file path that corresponds to 
# the T2W series of a specific patient, if exists.
# 
# -> path: path to the root folder of the patient
# <- d: absolute path to the t2w .mha image object

def fetch_patient_t2w(path):
  p_imgs = os.listdir(path)
  t2w_img = ""
  for i in range(len(p_imgs)):
    if p_imgs[i].endswith("t2w.mha"):
      t2w_img = path + p_imgs[i]
      break
  
  if t2w_img == "":
    print("Cannot find T2W series for patient: ", path)
  return t2w_img


# Fetch_patient_nii:
# ---------------------
# Retrieves the nifti file path that corresponds to 
# a specific patient, if exists.
# 
# -> path: path to the root folder of the patient
# <- d: absolute path to the nii object

def fetch_patient_nii(path, p_id, niis_dict):
  niis = os.listdir(path)
  nii = ""
  nii = niis_dict[p_id]

  if nii == "":
    print("Cannot find mask series for patient: ", p_id)
  return nii


# Build_data_list:
# ---------------------
# Builds a list that contains paths to the T2W MRI series
# and the corresponding niftis for all patients
# 
# -> paths: dictionary with all the preset paths
# -> niis_dict: dictionary for the masks of all patients
# <- imgs: list containing paths to the T2W MRI series
#    and the corresponding niftis

def build_data_list(paths, niis_dict):
  patients = os.listdir(paths['mha_root'])

  imgs = []

  print("Fetching MRI series and niftis for all patients...")
  for patient in patients:
    t2w_img = fetch_patient_t2w(paths['mha_root'] + '/' + patient + '/')
    nii = fetch_patient_nii(paths['nii_root'], patient, niis_dict)
    p = {
        'img_series': t2w_img,
        'mask_series': paths['nii_root'] + nii
    }
    imgs.append(p)
  return imgs


# series_to_images:
# ---------------------
# Converts the MRI series to standalone images. 
# 
# -> series_list: list containing paths to the T2W MRI series
#    and the corresponding niftis
# <- dataset: list with the standalone images paths and the 
#    corresponding niftis

def series_to_images(series_list):
  print("Converting series to standalone images...")
  print("This may take a while...")
  dataset = []
  for obj in series_list:
    mha, img_head = load(obj['img_series'])
    for i in range(mha.shape[2]):
      p = {
          'mha_path': obj['img_series'],
          'nii_path': obj['mask_series'],
          'channel': i
      }
      dataset.append(p)
  return dataset


def create_dataset(paths):

  niis_dict = build_dict_for_masks(paths['nii_root'])
  imgs      = build_data_list(paths, niis_dict)
  dataset   = series_to_images(imgs)
  return dataset