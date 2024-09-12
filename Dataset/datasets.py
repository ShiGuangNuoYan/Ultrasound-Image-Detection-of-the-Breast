from torch.utils.data import DataLoader, Dataset, random_split
import os
import cv2
import torch

class GAN_Dataset(Dataset):
    def __init__(self, opt,transform=None):
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.opt.data_path))

    def __getitem__(self, idx):
        img_name = os.listdir(self.opt.data_path)[idx]
        imgA = cv2.imread(self.opt.data_path + '/' + img_name)
        imgA = cv2.resize(imgA, (self.opt.image_scale_w, self.opt.image_scale_h))
        imgB = cv2.imread(self.opt.label_path + '/' + img_name[:-4] + '.png', 0)
        imgB = cv2.resize(imgB, (self.opt.image_scale_w, self.opt.image_scale_h))
        # imgB[imgB>30] = 255
        imgB = imgB / 255
        # imgB = imgB.astype('uint8')
        imgB = torch.FloatTensor(imgB)
        imgB = torch.unsqueeze(imgB, 0)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB

# 测试数据集
class test_Dataset(Dataset):
    # DATA_PATH = './test/img'
    # LABEL_PATH = './test/lab'
    def __init__(self, opt,transform=None):
        self.opt = opt
        self.transform = transform

    def __len__(self):
        return len(os.listdir('./munich/test/img'))

    def __getitem__(self, idx):
        img_name = os.listdir('./munich/test/img')
        img_name.sort(key=lambda x: int(x[:-4]))
        img_name = img_name[idx]
        imgA = cv2.imread('./munich/test/img' + '/' + img_name)
        imgA = cv2.resize(imgA, (self.opt.image_scale_w, self.opt.image_scale_h))
        imgB = cv2.imread('./munich/test/lab' + '/' + img_name[:-4] + '.png', 0)
        imgB = cv2.resize(imgB, (self.opt.image_scale_w, self.opt.image_scale_h))
        # imgB = imgB/255
        # imgB[imgB>30] = 255
        imgB = imgB / 255
        # imgB = imgB.astype('uint8')
        imgB = torch.FloatTensor(imgB)
        imgB = torch.unsqueeze(imgB, 0)
        # print(imgB.shape)
        if self.transform:
            # imgA = imgA/255
            # imgA = np.transpose(imgA, (2, 0, 1))
            # imgA = torch.FloatTensor(imgA)
            imgA = self.transform(imgA)
        return imgA, imgB, img_name[:-4]
