import os
from PIL import Image
from torch.utils.data import Dataset
from PIL import ImageOps

class FeatureDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 获取图像路径和标签路径
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        # 加载图像
        image = Image.open(image_path).convert('RGB')

        image = ImageOps.fit(image, (224, 224), method=Image.LANCZOS)

        # 如果有数据增强或预处理操作，应用在图像上
        if self.transform:
            image = self.transform(image)

        return image