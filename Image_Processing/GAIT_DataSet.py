import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch

# 自定义的步态数据集类，继承自 PyTorch 的 Dataset 类
class GAIT_DataSet(Dataset):
    def __init__(self, normal_root_dir, parkinsonian_root_dir, transform=None):
        # 正常样本数据的根目录
        self.normal_root_dir = normal_root_dir
        # 帕金森患者样本数据的根目录
        self.parkinsonian_root_dir = parkinsonian_root_dir
        # 图像预处理的转换操作
        self.transform = transform
        # 存储所有图像的文件路径
        self.image_paths = []
        # 存储所有图像对应的标签
        self.labels = []

        # 加载正常数据集
        # 使用 glob 模块获取正常样本目录下所有的 PNG 图像文件路径
        normal_paths = glob.glob(os.path.join(normal_root_dir, '*.png'))
        for path in normal_paths:
            # 将正常样本的图像路径添加到 image_paths 列表中
            self.image_paths.append(path)
            # 正常样本的标签设为 0
            self.labels.append(0)

        # 加载帕金森数据集
        # 使用 glob 模块获取帕金森患者样本目录下所有的 PNG 图像文件路径
        parkinsonian_paths = glob.glob(os.path.join(parkinsonian_root_dir, '*.png'))
        for path in parkinsonian_paths:
            # 将帕金森患者样本的图像路径添加到 image_paths 列表中
            self.image_paths.append(path)
            # 帕金森患者样本的标签设为 1
            self.labels.append(1)

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