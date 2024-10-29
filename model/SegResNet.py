from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.classifier import MultiClassifier, SingleClassifier

class ResNet50Model(nn.Module):
    def __init__(self, args, num_classes):
        super(ResNet50Model, self).__init__()

        self.resnet50 = models.resnet50()  # 不使用预训练权重
        self.args = args
        # 移除顶层的分类层
        self.resnet50.fc = nn.Identity()

        self.feature_classifier = MultiClassifier(num_classes)

    def forward(self, x):
        # ResNet50 提取特征
        x = self.resnet50(x)

        boundary_out, calcification_out, direction_out, shape_out = self.feature_classifier(x)
        return boundary_out, calcification_out, direction_out, shape_out
