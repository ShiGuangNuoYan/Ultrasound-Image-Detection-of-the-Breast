import torch.nn as nn

class SingleClassifier(nn.Module):
    def __init__(self,num_classes):
        super(SingleClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(2048)
        self.swish = nn.SiLU()

        # Dropout
        self.dropout = nn.Dropout(p=0.4)

        # 全连接层
        self.fc1 = nn.Linear(2048, 1024)  # 隐藏层有 1000 个单元
        self.fc2 = nn.Linear(1024, num_classes)  # 最终输出 n 个类别
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class MultiClassifier(nn.Module):
    def __init__(self, num_classes_per_feature):
        super(MultiClassifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(2048)
        self.swish = nn.SiLU(inplace=False)
        self.dropout = nn.Dropout(p=0.4,inplace=False)

        # 为每个特征创建独立的分类头
        self.boundary_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4,inplace=False),
            nn.Linear(1024, 2)
        )

        self.calcification_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4,inplace=False),
            nn.Linear(1024, 2)
        )

        self.direction_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4,inplace=False),
            nn.Linear(1024, 2)
        )

        self.shape_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.SiLU(inplace=False),
            nn.Dropout(p=0.4,inplace=False),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.view(x.size(0), -1)  # 展平

        # 通过每个特征的分类头进行独立分类
        boundary_out = self.boundary_head(x)
        calcification_out = self.calcification_head(x)
        direction_out = self.direction_head(x)
        shape_out = self.shape_head(x)

        return boundary_out, calcification_out, direction_out, shape_out