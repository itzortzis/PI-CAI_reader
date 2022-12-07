import csv
import numpy as np
from skimage import exposure
from skimage.filters import gaussian, laplace


# Build_dcm_dict:
# ---------------
# Given the descriptive csv file of the original
# INBreast dataset, this function generates a dictionary
# with the needed information for each image
# 
# -> path_to_csv: path to the csv file
# <- Dictionary containing the corresponding info

def build_dcm_dict(path_to_csv):
  dcm_list = []

  with open(path_to_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    dict_key   = 0

    for row in csv_reader:
      if line_count == 0:
        line_count += 1
        continue
      obj = {"filename": row[5],
              "acr": row[6],
              "birads": row[7],
              "mass": row[8],
              "calc": row[9]}
      dcm_list.append(obj)
      line_count += 1
      dict_key += 1
    print(f'Processed {line_count} lines.')

  return dcm_list


# Extract_mal_dcms:
# ---------------
# Given a list containing info about each DICOM file of
# the INBreast dataset, this function extract only the 
# images that contain some kind of malignacy
# 
# -> dcm_list: list containing info about the DICOM files
# <- non_h: dictionary containing the file names of the 
#    extracted images

def extract_mal_dcms(dcm_list):
  non_h = []
  for i in range(len(dcm_list)):
    if dcm_list[i]['mass'] == 'X' or dcm_list[i]['calc'] == "X":
      non_h.append(dcm_list[i]['filename'])
  return non_h


# Img_augmentation:
# -----------------
# Generates augmentations for the given image
# 
# -> img: numpy array with the image information
# -> mask: numpy array with the annotation mask info
# <- augs: list containing the generated augmentations

def img_augmentation(img, mask):
  augs = []
  for i in range(4):
    if i == 0:
      aug_img  = gaussian(img, sigma=0.9, multichannel=False)
      aug_mask = mask
    if i == 1:
      aug_img  = exposure.adjust_gamma(img, 2)
      aug_mask = mask
    if i == 2:
      aug_img  = exposure.equalize_hist(img)
      aug_mask = mask
    if i == 3:
      aug_img  = np.fliplr(img)
      aug_mask = np.fliplr(mask)
    
    if np.sum(mask[:, :, 0]) == 0 and np.sum(mask[:, :, 1]) == 0: cat = "H"
    if np.sum(mask[:, :, 0]) != 0 and np.sum(mask[:, :, 1]) == 0: cat = "T"
    if np.sum(mask[:, :, 0]) == 0 and np.sum(mask[:, :, 1]) != 0: cat = "C"
    if np.sum(mask[:, :, 0]) != 0 and np.sum(mask[:, :, 1]) != 0: cat = "TC" 

    augs.append({
        "img": aug_img,
        "mask": aug_mask,
        "cat": cat
    })

  return augs


# Crop_image:
# -----------
# Applies a peripheral cropping to the given image,
# aiming to get rid of non-essential areas with zero
# valued pixels
# 
# -> img: 2d numpy array with the image information
# <- img: the cropped image numpy array
# <- (.): tuple with the limits of cropping

def crop_image(img):
  l = 0
  r = img.shape[1]-1
  t = 0
  d = img.shape[0]-1
  
  while img[:, l].sum() == 0: l += 1
  while img[:, r].sum() == 0: r -= 1
  while img[t, :].sum() == 0: t += 1
  while img[d, :].sum() == 0: d -= 1
  
  return img[t:d, l:r], (l, r, t, d)


# Extract_img_cat:
# ----------------
# Defines the image category according to the csv 
# file information
# 
# -> img_obg: retrieved row from the csv file
# <- cat: The image category;
#     * 0 -- healthy breast
#     * 1 -- breast contains mass
#     * 2 -- breast contains calcifications
#     * 3 -- breast contains both masses and calcs

def extract_img_cat(img_obj):
  has_calc = img_obj["calc"] == 'X'
  has_mass = img_obj["mass"] == 'X'

  if not has_calc and not has_mass: cat = 0
  if not has_calc and     has_mass: cat = 1
  if has_calc     and not has_mass: cat = 2
  if has_calc     and     has_mass: cat = 3

  return cat

