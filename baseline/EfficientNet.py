import argparse
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset.ImageDataset import ImageDataset
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 设置图像的预处理（包括归一化等）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 定义 EfficientNet-B0 模型
class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB0Model, self).__init__()

        # 加载 EfficientNet-B0 模型，使用 ImageNet 的预训练权重
        self.efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # 移除顶层的分类层
        self.efficientnet_b0.classifier = nn.Identity()

        # Batch Normalization + Swish Activation
        self.batch_norm = nn.BatchNorm2d(1280)  # EfficientNet-B0 输出通道数为 1280
        self.swish = nn.SiLU()  # Swish 激活函数

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # 全连接层
        self.fc1 = nn.Linear(1280, 1000)  # 隐藏层有 1000 个单元
        self.fc2 = nn.Linear(1000, num_classes)  # 最终输出 6 个类别

    def forward(self, x):
        # EfficientNet-B0 提取特征
        x = self.efficientnet_b0.features(x)

        # Batch Normalization + Swish 激活函数
        x = self.batch_norm(x)
        x = self.swish(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平

        # Dropout + 全连接层
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.relu(x)

        # 输出层
        x = self.fc2(x)
        return x  # 这里不再需要 softmax，因为 CrossEntropyLoss 会处理


# 图像预处理函数
def prepare_image(file):
    img = Image.open(file).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # EfficientNet 标准预处理
    ])
    img_tensor = transform(img)
    return img_tensor


# 训练循环
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    print(">>>>>>>>>>>>>>> training time <<<<<<<<<<<<<<<<<<<<<<")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%')



# 测试循环并记录到 TensorBoard
def evaluate_model(model, test_loader, criterion, device, writer, epoch,best_loss):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    print(">>>>>>>>>>>>>>> testing time <<<<<<<<<<<<<<<<<<<<<<")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    # 打印测试结果
    print(f'Test Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.2f}%')

    # 保存最佳模型
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), f'./save/{epoch+1}_best_model.pth')
        print(f'Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}')

    # 将 loss 和 accuracy 记录到 TensorBoard
    writer.add_scalar('Loss/test', epoch_loss, epoch)
    writer.add_scalar('Accuracy/test', accuracy, epoch)
    return best_loss


# 定义数据集和数据加载器
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = args.image_dir
    label_dir = args.label_dir
    test_image_dir = args.test_dir
    test_label_dir = args.test_label_dir

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir='./runs/efficientnet_experiment')
    best_loss = float('inf')
    # 模型实例化并移动到GPU
    model = EfficientNetB0Model(num_classes=6).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建训练和测试数据集
    train_dataset = ImageDataset(image_dir=image_dir, label_dir=label_dir, transform=transform)
    test_dataset = ImageDataset(image_dir=test_image_dir, label_dir=test_label_dir, transform=transform)

    # 创建 DataLoader，设置批量大小和是否打乱数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # 训练和评估
    epochs = args.epoch
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}:')
        train_model(model, train_loader, criterion, optimizer, device)
        best_loss = evaluate_model(model, test_loader, criterion, device, writer, epoch,best_loss)
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=False, type=str,default="E:/Dataset/Contest4/Ultrasound_breast_data/classification_train/train/All/images", help="input train image dir")
    parser.add_argument("--label_dir", required=False, type=str,default="E:/Dataset/Contest4/Ultrasound_breast_data/classification_train/train/All/labels", help="input train label dir")
    parser.add_argument("--test_dir", required=False, type=str, default="E:/Dataset/Contest4/Ultrasound_breast_data/test_A/classification-test/A/All/images",help="input test image dir")
    parser.add_argument("--test_label_dir", required=False, default="E:/Dataset/Contest4/Ultrasound_breast_data/test_A/classification-test/A/All/labels",type=str, help="input test label dir")
    parser.add_argument("--epoch", required=False, type=int, default=60 ,help="epoch")
    parser.add_argument("--lr",required=False, type=float, default=0.001, help="learning rate")
    args = parser.parse_args()
    main(args)

