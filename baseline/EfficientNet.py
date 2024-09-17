import argparse
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset.ImageDataset import ImageDataset,FeatureDataset
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from util.loss import CustomCrossEntropyLoss

# 设置图像的预处理（包括归一化等）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 指定GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SingleClassifier(nn.Module):
    def __init__(self,num_classes):
        super(SingleClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1280)
        self.swish = nn.SiLU()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)


        # Dropout
        self.dropout = nn.Dropout(p=0.4)

        # 全连接层
        self.fc1 = nn.Linear(1280, 1000)  # 隐藏层有 1000 个单元
        self.fc2 = nn.Linear(1000, num_classes)  # 最终输出 n 个类别
    def forward(self, x):
        x = self.batch_norm(x)

        x = self.swish(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MultiClassifier(nn.Module):
    def __init__(self,total_classes):
        super(MultiClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm2d(1280)
        self.swish = nn.SiLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4)
        num_classes = sum(total_classes)
        self.num_classes_per_feature = total_classes
        # 全连接层
        self.fc1 = nn.Linear(1280, 1000)  # 隐藏层有 1000 个单元
        self.fc2 = nn.Linear(1000, num_classes)  # 最终输出 n 个类别

    def forward(self,x):
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1) #展平
        x = self.dropout(x)
        x = self.fc1(x)
        outputs = self.fc2(x)

        start = 0
        output_splits = []
        for num_classes in self.num_classes_per_feature:
            output_splits.append(outputs[:, start:start + num_classes])
            start += num_classes
        # 输出4个特征的分类结果
        boundary_out, calcification_out, direction_out, shape_out = output_splits
        return boundary_out, calcification_out, direction_out, shape_out


# 定义 EfficientNet-B0 模型
class EfficientNetB0Model(nn.Module):
    def __init__(self, args,num_classes):
        super(EfficientNetB0Model, self).__init__()

        # 加载 EfficientNet-B0 模型，使用 ImageNet 的预训练权重
        self.efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.args = args
        # 移除顶层的分类层
        self.efficientnet_b0.classifier = nn.Identity()
        if not args.f:
            self.image_classifier = SingleClassifier(num_classes)
        else:
            self.feature_classifier = MultiClassifier(num_classes)

    def forward(self, x):
        # EfficientNet-B0 提取特征
        x = self.efficientnet_b0.features(x)
        if not self.args.f:
            x = self.image_classifier(x)
            return x  # 这里不再需要 softmax，因为 CrossEntropyLoss 会处理
        else:
            boundary_out, calcification_out, direction_out, shape_out = self.feature_classifier(x)
            return boundary_out, calcification_out, direction_out, shape_out

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
def train_model(model, train_loader, criterion, optimizer, device, is_feature):
    if not is_feature:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(">>>>>>>>>>>>>>>image training time <<<<<<<<<<<<<<<<<<<<<<")
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

    else:
        running_loss = 0.0
        b_correct = 0
        c_correct = 0
        d_correct = 0
        s_correct = 0
        total = 0
        print(">>>>>>>>>>>>>>>feature training time <<<<<<<<<<<<<<<<<<<<<<")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            boundary_out, calcification_out, direction_out, shape_out = model(images)

            boundary_loss = criterion(boundary_out, labels[:,0])
            calcification_loss = criterion(calcification_out, labels[:,1])
            direction_loss = criterion(direction_out, labels[:,2])
            shape_loss = criterion(shape_out, labels[:,3])
            loss = boundary_loss + calcification_loss + direction_loss + shape_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, b_predicted = torch.max(boundary_out, 1)
            b_correct += (b_predicted == labels[:,0]).sum().item()

            _, c_predicted = torch.max(calcification_out, 1)
            c_correct += (c_predicted == labels[:, 1]).sum().item()

            _, d_predicted = torch.max(direction_out, 1)
            d_correct += (d_predicted == labels[:, 2]).sum().item()

            _, s_predicted = torch.max(shape_out, 1)
            s_correct += (s_predicted == labels[:, 3]).sum().item()

            total += labels.size(0)


        epoch_loss = running_loss / len(train_loader)
        b_accuracy = 100 * b_correct / total
        c_accuracy = 100 * c_correct / total
        d_accuracy = 100 * d_correct / total
        s_accuracy = 100 * s_correct / total
        print(f'Train Loss: {epoch_loss:.4f}, Boundary Train Accuracy: {b_accuracy:.2f}%')
        print(f'Train Loss: {epoch_loss:.4f}, Calcification Train Accuracy: {c_accuracy:.2f}%')
        print(f'Train Loss: {epoch_loss:.4f}, Direction Train Accuracy: {d_accuracy:.2f}%')
        print(f'Train Loss: {epoch_loss:.4f}, Shape Train Accuracy: {s_accuracy:.2f}%')


# 测试循环并记录到 TensorBoard
def evaluate_model(model, test_loader, criterion, device, writer, epoch,best_loss,save_path,args):
    model.eval()
    if not args.f:
        running_loss = 0.0
        correct = 0
        total = 0
        print(">>>>>>>>>>>>>>>image testing time <<<<<<<<<<<<<<<<<<<<<<")
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
        print(f'Image Test Loss: {epoch_loss:.4f}, Image Test Accuracy: {accuracy:.2f}%')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.efficientnet_b0.state_dict(), save_path + '/best_model.pth')  # 只保留主干网络的参数？
            print(f'Best model saved at epoch {epoch + 1} with loss {best_loss:.4f}')

        # 将 loss 和 accuracy 记录到 TensorBoard
        writer.add_scalar('Loss/test', epoch_loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)
        return best_loss
    else:
        running_loss = 0.0
        b_correct = 0
        c_correct = 0
        d_correct = 0
        s_correct = 0
        total = 0
        print(">>>>>>>>>>>>>>>feature test time <<<<<<<<<<<<<<<<<<<<<<")
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            boundary_out, calcification_out, direction_out, shape_out = model(images)

            boundary_loss = criterion(boundary_out, labels[:,0])
            calcification_loss = criterion(calcification_out, labels[:,1])
            direction_loss = criterion(direction_out, labels[:,2])
            shape_loss = criterion(shape_out, labels[:,3])

            loss = boundary_loss + calcification_loss + direction_loss + shape_loss
            running_loss += loss.item()

            _, b_predicted = torch.max(boundary_out, 1)
            b_correct += (b_predicted == labels[:,0]).sum().item()

            _, c_predicted = torch.max(calcification_out, 1)
            c_correct += (c_predicted == labels[:, 1]).sum().item()

            _, d_predicted = torch.max(direction_out, 1)
            d_correct += (d_predicted == labels[:, 2]).sum().item()

            _, s_predicted = torch.max(shape_out, 1)
            s_correct += (s_predicted == labels[:, 3]).sum().item()

            total += labels.size(0)


        epoch_loss = running_loss / len(test_loader)
        b_accuracy = 100 * b_correct / total
        c_accuracy = 100 * c_correct / total
        d_accuracy = 100 * d_correct / total
        s_accuracy = 100 * s_correct / total
        print(f'Feature Test Loss: {epoch_loss:.4f}, Boundary test Accuracy: {b_accuracy:.2f}%')
        print(f'Feature Test Loss: {epoch_loss:.4f}, Calcification test Accuracy: {c_accuracy:.2f}%')
        print(f'Feature Test Loss: {epoch_loss:.4f}, Direction test Accuracy: {d_accuracy:.2f}%')
        print(f'Feature Test Loss: {epoch_loss:.4f}, Shape test Accuracy: {s_accuracy:.2f}%')


# 定义数据集和数据加载器
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = args.epoch
    if not args.f:
        train_image = args.image_dir + '/train/images'
        train_label = args.image_dir + '/train/labels'
        test_image = args.image_dir + '/test/images'
        test_label = args.image_dir + '/test/labels'

        # 创建 TensorBoard writer
        writer = SummaryWriter(log_dir='./runs/efficientnet_image_classification')
        best_loss = float('inf')
        # 模型实例化并移动到GPU
        model = EfficientNetB0Model(args,num_classes=6).to(device)

        # 损失函数和优化器
        criterion = CustomCrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # 创建训练和测试数据集
        train_dataset = ImageDataset(image_dir=train_image, label_dir=train_label, transform=transform)
        test_dataset = ImageDataset(image_dir=test_image, label_dir=test_label, transform=transform)

        # 创建 DataLoader，设置批量大小和是否打乱数据
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

        # 训练和评估
        for epoch in range(total_epochs):
            print(f'Epoch {epoch + 1}/{total_epochs} in image train:')
            train_model(model, train_loader, criterion, optimizer, device,args.f)
            best_loss = evaluate_model(model, test_loader, criterion, device, writer, epoch, best_loss,args.save_path,args)
        writer.close()
    else:
        train_image = args.feature_dir + '/train/images'
        train_label = args.feature_dir + '/train/labels'
        test_image = args.feature_dir + '/test/images'
        test_label = args.feature_dir + '/test/labels'

        # 创建 TensorBoard writer
        writer = SummaryWriter(log_dir='./runs/efficientnet_feature_classification')
        best_loss = float('inf')

        # 损失函数和优化器
        criterion = CustomCrossEntropyLoss()

        # 创建训练和测试数据集
        train_dataset = FeatureDataset(image_dir=train_image, label_dir=train_label, transform=transform)
        test_dataset = FeatureDataset(image_dir=test_image, label_dir=test_label, transform=transform)

        # 创建 DataLoader，设置批量大小和是否打乱数据
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

        # 模型实例化并移动到GPU
        model = EfficientNetB0Model(args,num_classes=[2,2,2,2]).to(device)
        model_path = args.save_path + '/best_model.pth'

        if os.path.exists(model_path):
            model.efficientnet_b0.load_state_dict(torch.load(model_path))
            model.efficientnet_b0.eval()
            model.feature_classifier.train()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            for epoch in range(total_epochs):
                print(f'Epoch {epoch + 1}/{total_epochs} in feature train:')
                train_model(model, train_loader, criterion, optimizer, device,args.f)
                best_loss = evaluate_model(model, test_loader, criterion, device, writer, epoch, best_loss,
                                           args.save_path,args)
            writer.close()

        else:
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            model.train()
            for epoch in range(total_epochs):
                print(f'Epoch {epoch + 1}/{total_epochs} in feature train:')
                train_model(model, train_loader, criterion, optimizer, device,args.f)
                best_loss = evaluate_model(model, test_loader, criterion, device, writer, epoch, best_loss,
                                           args.save_path,args)
            writer.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--image_dir", required=False, type=str,default="E:/Dataset/Contest4/Ultrasound_breast_data/classification_train/train/All/images", help="input train image dir")
    # parser.add_argument("--image_label", required=False, type=str,default="E:/Dataset/Contest4/Ultrasound_breast_data/classification_train/train/All/labels", help="input train label dir")
    # parser.add_argument("--test_dir", required=False, type=str, default="E:/Dataset/Contest4/Ultrasound_breast_data/test_A/classification-test/A/All/images",help="input test image dir")
    # parser.add_argument("--test_image_label", required=False, default="E:/Dataset/Contest4/Ultrasound_breast_data/test_A/classification-test/A/All/labels",type=str, help="input test label dir")

    parser.add_argument("--image_dir",required=False, type=str, help="input image classification dir")
    parser.add_argument("--feature_dir",required=False, type=str, help="input feature classification dir")
    parser.add_argument("--epoch", required=False, type=int, default=80 ,help="epoch")
    parser.add_argument("--lr",required=False, type=float, default=0.001, help="learning rate")
    parser.add_argument("--f",required=False,type=bool,default=False,help='is feature classification')
    parser.add_argument("--save_path",required=False,type=str,default='./save',help='best backbone model save path')
    args = parser.parse_args()
    main(args)

