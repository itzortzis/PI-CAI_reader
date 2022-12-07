import torch
import cv2

class Dataset(torch.utils.data.Dataset):
  def __init__(self, d):
    self.dtst = d

  def __len__(self):
    return len(self.dtst)

  def __getitem__(self, index):
    p = self.dtst[index]['mha_path']
    img, img_head = load(p)
    img = np.array(img, dtype=np.float32)
    img = img[:, :, self.dtst[index]['channel']]
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    x = torch.from_numpy(img)
    

    nii = nib.load(self.dtst[index]['nii_path'])
    nii = nii.get_fdata()
    nii = nii[:, :, self.dtst[index]['channel']]
    nii = cv2.resize(nii, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    y = torch.from_numpy(nii)
    

    return x, y