import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=999):
        super(CustomCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index  # 忽略的标签值

    def forward(self, outputs, labels):
        # 找出所有不是 ignore_index 的位置
        valid_mask = (labels != self.ignore_index)

        # 如果没有有效的标签，返回 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True).to(outputs.device)

        # 只计算有效标签的损失
        filtered_outputs = outputs[valid_mask]
        filtered_labels = labels[valid_mask]

        # 使用 F.cross_entropy 计算损失
        loss = F.cross_entropy(filtered_outputs, filtered_labels)

        return loss
