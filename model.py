import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 低级特征提取器模型类
class LowLevelFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一个卷积层，输入通道为 1，输出通道为 16，卷积核大小为 5，填充为 2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        # GELU 激活函数
        self.gelu1 = nn.GELU()
        # LayerNorm 层，对卷积层的输出进行归一化
        self.ln1 = nn.LayerNorm([16, 224, 224])

        # 添加 Dropout 层
        self.dropout = nn.Dropout(0.5)

        # 第二个卷积层，输入通道为 16，输出通道为 32，卷积核大小为 5，填充为 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.gelu2 = nn.GELU()
        self.ln2 = nn.LayerNorm([32, 224, 224])

        # 第三个卷积层，输入通道为 32，输出通道为 64，卷积核大小为 3，填充为 1
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.gelu3 = nn.GELU()
        self.ln3 = nn.LayerNorm([64, 224, 224])

        # 全局平均池化层，将卷积特征转换为一维向量
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层，输入维度为 64，输出维度为 1
        self.fc = nn.Linear(64, 1)
        # Sigmoid 激活函数，将输出转换为概率值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一个卷积层 -> GELU 激活 -> LayerNorm
        x = self.ln1(self.gelu1(self.conv1(x)))
        x = self.dropout(x)
        # 第二个卷积层 -> GELU 激活 -> LayerNorm
        x = self.ln2(self.gelu2(self.conv2(x)))
        # 第三个卷积层 -> GELU 激活 -> LayerNorm
        x = self.ln3(self.gelu3(self.conv3(x)))
        # 全局平均池化
        x = self.gap(x)
        # 将张量展平为一维向量
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.fc(x)
        # Sigmoid 激活
        x = self.sigmoid(x)
        return x

# 高级特征提取器模型类
class HigherFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 ImageNet 预训练的 ResNet50 模型
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # 修改 ResNet 的第一个卷积层，使其输入通道为 1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉 ResNet 的最后一层全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Transformer 编码层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8, batch_first=True),
            num_layers=2
        )
        # 全连接层，输入维度为 2048，输出维度为 1
        self.fc = nn.Linear(2048, 1)
        # Sigmoid 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通过 ResNet 提取特征
        x = self.resnet(x)
        # 将张量展平为一维向量
        x = x.view(x.size(0), -1)
        # 保存残差
        residual = x
        # 调整维度以适应 Transformer 输入
        x = x.unsqueeze(1)
        # 通过 Transformer 编码层
        x = self.transformer(x)
        # 去掉多余的维度
        x = x.squeeze(1)
        # 残差连接
        x = x + residual
        # 全连接层
        x = self.fc(x)
        # Sigmoid 激活
        x = self.sigmoid(x)
        return x

# 特征融合模型类
class FeatureFusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 低级特征提取器
        self.low_level_extractor = LowLevelFeatureExtractor()
        # 高级特征提取器
        self.high_level_extractor = HigherFeatureExtractor()
        # 假设低级和高级特征提取器的输出维度都为 1，合并后的维度为 2
        combined_dim = 2
        # 增加全连接层
        self.fc1 = nn.Linear(combined_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        # 提取低级特征
        low_level_feature = self.low_level_extractor(x).squeeze()
        # 提取高级特征
        high_level_feature = self.high_level_extractor(x).squeeze()

        # 特征融合，将低级和高级特征在维度 1 上拼接
        fused_feature = torch.cat((low_level_feature.unsqueeze(1), high_level_feature.unsqueeze(1)), dim=1)

        # 通过新增的全连接层
        x = self.relu1(self.fc1(fused_feature))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x