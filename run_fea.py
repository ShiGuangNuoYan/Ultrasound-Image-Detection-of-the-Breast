import torch
import os
import pandas as pd
from tqdm import tqdm
from model.SegResNet import ResNet50Model
from dataset.ImageDataset import FeatureDataset
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms

# 设置图像的预处理（包括归一化等）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def evaluate_model(model, test_loader, device):
    model.eval()
    total = 0
    all_b_predictions = []
    all_c_predictions = []
    all_d_predictions = []
    all_s_predictions = []
    print(">>>>>>>>>>>>>>>feature test time <<<<<<<<<<<<<<<<<<<<<<")
    for images in tqdm(test_loader):
        images= images.to(device)
        boundary_out, calcification_out, direction_out, shape_out = model(images)
        _, b_predicted = torch.max(boundary_out, 1)

        _, c_predicted = torch.max(calcification_out, 1)

        _, d_predicted = torch.max(direction_out, 1)

        _, s_predicted = torch.max(shape_out, 1)

        total += images.size(0)
        all_b_predictions.extend(b_predicted.cpu().numpy())
        all_c_predictions.extend(c_predicted.cpu().numpy())
        all_d_predictions.extend(d_predicted.cpu().numpy())
        all_s_predictions.extend(s_predicted.cpu().numpy())


    # 创建id列
    id_column = list(range(1, total + 1))

    # 将每列的预测结果保存到 CSV 文件
    df = pd.DataFrame({
        'id': id_column,
        'boundary': all_b_predictions,
        'calcification': all_c_predictions,
        'direction': all_d_predictions,
        'shape': all_s_predictions
    })
    df.to_csv('./fea_pre.csv', index=False)
    return


def run_feature(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建训练和测试数据集
    test_dataset = FeatureDataset(image_dir=args.feature_dir , transform=transform)

    # 创建 DataLoader，设置批量大小和是否打乱数据
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    # 模型实例化并移动到GPU
    model = ResNet50Model(args, num_classes=[2, 2, 2, 2]).to(device)

    if os.path.exists(args.fea_model_path):
        model.load_state_dict(torch.load(args.fea_model_path))
        model.eval()
        evaluate_model(model, test_loader, device)