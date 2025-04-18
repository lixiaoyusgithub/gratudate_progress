import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch

# 自定义的步态数据集类
class GAIT_DataSet(Dataset):
    def __init__(self, normal_json_path, parkinsonian_json_path, transform=None):
        # 正常样本的 JSON 文件路径
        self.normal_json_path = normal_json_path
        # 帕金森患者样本的 JSON 文件路径
        self.parkinsonian_json_path = parkinsonian_json_path
        # 图像预处理的转换操作
        self.transform = transform
        # 存储所有图像的文件路径
        self.image_paths = []
        # 存储所有图像对应的标签
        self.labels = []

        # 加载正常数据集
        if os.path.exists(self.normal_json_path):
            try:
                with open(self.normal_json_path, 'r') as f:
                    normal_data = json.load(f)
                    self.image_paths.extend(normal_data["image_paths"])
                    self.labels.extend(normal_data["labels"])
            except Exception as e:
                print(f"读取正常样本 JSON 文件时出错: {e}")
        else:
            print(f"正常样本 JSON 文件 {self.normal_json_path} 不存在。")

        # 加载帕金森数据集
        if os.path.exists(self.parkinsonian_json_path):
            try:
                with open(self.parkinsonian_json_path, 'r') as f:
                    parkinsonian_data = json.load(f)
                    self.image_paths.extend(parkinsonian_data["image_paths"])
                    self.labels.extend(parkinsonian_data["labels"])
            except Exception as e:
                print(f"读取帕金森患者样本 JSON 文件时出错: {e}")
        else:
            print(f"帕金森患者样本 JSON 文件 {self.parkinsonian_json_path} 不存在。")

    def __len__(self):
        # 返回数据集的样本数量，即图像路径列表的长度
        return len(self.image_paths)

    def __getitem__(self, item):
        # 根据索引获取图像的文件路径
        img_path = self.image_paths[item]
        # 根据索引获取图像对应的标签
        label = self.labels[item]
        # 打开图像并将其转换为灰度图
        image = Image.open(img_path).convert('L')
        # 如果定义了图像预处理转换操作，则对图像进行处理
        if self.transform:
            image = self.transform(image)
        # 将标签转换为 PyTorch 的张量
        label = torch.tensor(label, dtype=torch.long)
        # 返回处理后的图像和对应的标签
        return image, label