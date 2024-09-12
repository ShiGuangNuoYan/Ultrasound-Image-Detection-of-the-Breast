import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 自定义 Dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像路径和标签路径
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('_cropped.jpg', '.txt'))  # 假设标签是 .txt 文件

        # 加载图像
        image = Image.open(image_path).convert('RGB')

        # 加载标签（假设标签是一个简单的整数）
        with open(label_path, 'r') as f:
            label = int((f.read().strip().split(' '))[0])

        # 如果有数据增强或预处理操作，应用在图像上
        if self.transform:
            image = self.transform(image)

        return image, label