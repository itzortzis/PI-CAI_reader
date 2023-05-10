from medpy.io import load
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import cv2



class PicaiReader():

	def __init__(self):
		pass


	# Build_dict_for_masks:
	# ---------------------
	# This function creates a dictionary for the
	# nifti files locations assigning them to the
	# corresponding patients ids
	#
	# -> path: path to the root folder of nifi files
	# <- d: dictionary with the nii locations

	def build_dict_for_masks(self, path):
		print("Building dictionary for masks...")
		self.niis_dict = {}
		niis_names = os.listdir(path)

		for nii in niis_names:
			tokens = nii.split('_')
			p_id = tokens[0]
			self.niis_dict[p_id] = nii


	# Fetch_patient_t2w:
	# ---------------------
	# Retrieves the file path that corresponds to
	# the T2W series of a specific patient, if exists.
	#
	# -> path: path to the root folder of the patient
	# <- d: absolute path to the t2w .mha image object

	def fetch_patient_t2w(self, path):
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

	def fetch_patient_nii(self, path, p_id):
		niis = os.listdir(path)
		nii = ""
		found = True

		try:
			nii = self.niis_dict[p_id]
		except:
			print("Cannot find mask series for patient: ", p_id)
			found = False

		return nii, found


	# Build_data_list:
	# ---------------------
	# Builds a list that contains paths to the T2W MRI series
	# and the corresponding niftis for all patients
	#
	# -> paths: dictionary with all the preset paths
	# -> niis_dict: dictionary for the masks of all patients
	# <- imgs: list containing paths to the T2W MRI series
	#    and the corresponding niftis

	def build_mri_list(self, paths):
		patients = os.listdir(paths['mha_root'])

		self.mris = []

		print("Fetching MRI series and niftis for all patients...")
		w = 0
		for patient in patients:
			if w > 15:
				break
			w += 1
			if patient.startswith("."):
				continue
			path_to_patient = paths['mha_root'] + '/' + patient + '/'
			t2w_img = self.fetch_patient_t2w(path_to_patient)
			nii, found = self.fetch_patient_nii(paths['nii_root'], patient)
			if not found:
				continue
			p = {
			    'img_series': t2w_img,
			    'mask_series': paths['nii_root'] + nii
			}
			self.mris.append(p)
		print(len(self.mris), " patients were successfully fetched.")


	def split_mris(self):
		train_len = int(0.7 * len(self.mris))
		valid_len = int(0.2 * len(self.mris))
		test_len  = int(0.1 * len(self.mris))

		train_start = 0
		train_end   = train_len
		valid_start = train_end
		valid_end   = valid_start + valid_len
		test_start  = valid_end
		test_end    = test_start + test_len

		self.train_mris = self.mris[train_start: train_end]
		self.valid_mris = self.mris[valid_start: valid_end]
		self.test_mris  = self.mris[test_start: test_end]


	def load_obj(self, obj):
		mri, img_head = load(obj['mha_path'])
		nii = nib.load(obj['nii_path'])
		nii = nii.get_fdata()

		return mri, nii


	def resize_obj(self, obj):
		# obj = (obj - np.min(obj)) / (np.max(obj) - np.min(obj))
		obj = cv2.resize(obj, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

		return obj


	def is_obj_valid(self, mri, nii):
		if mri.shape[2] != nii.shape[2]:
			return False
		return True


	def list_to_numpy(self, l):
		slice_count = self.count_slices(l)
		n_a = np.zeros((slice_count, 256, 256, 2))
		w = 0
		for i in range(len(l)):
			mri, nii = self.load_obj(l[i])
			if not self.is_obj_valid(mri, nii):
				continue
			for j in range(mri.shape[2]):
				print(mri[:, :, j])
				slice = self.resize_obj(mri[:, :, j])
				print(slice)
				n_a[w, :, :, 0] = slice
				slice = self.resize_obj(nii[:, :, j])
				n_a[w, :, :, 1] = slice
				w += 1
		n_a = n_a[:w, :, :, :]
		return n_a



	def count_slices(self, l):
		count = 0
		for i in range(len(l)):
			mri, img_head = load(l[i]['mha_path'])
			count += mri.shape[2]
		return count

	# series_to_images:
	# ---------------------
	# Converts the MRI series to standalone images.
	#
	# -> series_list: list containing paths to the T2W MRI series
	#    and the corresponding niftis
	# <- dataset: list with the standalone images paths and the
	#    corresponding niftis

	def mris_to_s_paths(self, series_list):
		print("Converting MRI scans to standalone images...")
		print("This may take a while...")
		imgs = []
		for obj in series_list:
			mha, img_head = load(obj['img_series'])
			for i in range(mha.shape[2]):

				p = {
					'mha_path': obj['img_series'],
					'nii_path': obj['mask_series'],
					'slice': i
				}
				imgs.append(p)
		return imgs


	def create_sets_lists(self):
		self.train_set_l = self.mris_to_s_paths(self.train_mris)
		self.valid_set_l = self.mris_to_s_paths(self.valid_mris)
		self.test_set_l  = self.mris_to_s_paths(self.test_mris)

		print(len(self.train_set_l), len(self.valid_set_l), len(self.test_set_l))


	def create_dataset(self, paths):

		self.build_dict_for_masks(paths['nii_root'])
		self.build_mri_list(paths)
		self.split_mris()
		self.create_sets_lists()
		self.test_set = self.list_to_numpy(self.test_set_l)


paths = {
	'nii_root': '/home/itzo/datasets/picai/picai_labels/csPCa_lesion_delineations/human_expert/resampled/',
	'mha_root': '/home/itzo/datasets/picai/fold0'
}

d = PicaiReader()
d.create_dataset(paths)

plt.figure()
plt.imshow(d.test_set[0, :, :, 0], cmap='gray')
plt.savefig('fig.png')

plt.figure()
plt.imshow(d.test_set[0, :, :, 1], cmap='gray')
plt.savefig('fig_mask.png')
