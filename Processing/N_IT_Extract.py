import json

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import glob

class N_IT_Extract(Dataset):
    def __init__(self, root_dir, transform=None, data_type='silhouettes', view_type='_back'):
        # 存储数据集的根目录路径，后续将基于此路径查找数据文件
        self.root_dir = root_dir
        # 存储图像转换操作，例如数据增强、归一化等，默认为 None
        self.transform = transform
        # 用于存储图像文件的路径，方便后续根据索引获取图像
        self.image_paths = []
        # 用于存储每个图像对应的标签，这里标签统一为 1
        self.labels = []
        # 存储要使用的数据类型，默认为 'GEIs',silhouettes
        self.data_type = data_type
        # 存储要使用的视图类型，默认为 '_front',_back
        self.view_type = view_type

        # 构建 GAIT-IT 目录读取路径，使用用户选择的数据类型
        # 路径格式为 root_dir/Parkinson/*/data_type/*
        path_class_it = os.path.join(root_dir, 'Normal', '*', self.data_type, '*')
        # 使用 glob.glob 函数查找符合路径规则的所有文件夹
        folders = glob.glob(path_class_it)
        # 遍历找到的所有文件夹
        for folder in folders:
            # 遍历当前文件夹下的子目录
            sub_folders = os.listdir(folder)
            # 遍历每个子目录
            for sub_folder in sub_folders:
                # 检查子目录名是否以用户选择的视图类型结尾
                if sub_folder.endswith(self.view_type):
                    # 构建子目录的完整路径
                    sub_folder_path = os.path.join(folder, sub_folder)
                    # 遍历子目录下的所有文件
                    for file in os.listdir(sub_folder_path):
                        # 检查文件扩展名是否为图片格式（png、jpg、jpeg）
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # 构建图片文件的完整路径
                            img_path = os.path.join(sub_folder_path, file)
                            # 将图片路径添加到 image_paths 列表中
                            self.image_paths.append(img_path)
                            # 为该图片添加标签 1，表示该图片属于特定类别
                            self.labels.append(1)

    # 定义获取数据集长度的方法，返回 image_paths 列表的长度
    def __len__(self):
        return len(self.image_paths)

    # 定义根据索引获取数据集中单个样本的方法
    def __getitem__(self, item):
        # 根据索引获取图片文件的路径
        img_path = self.image_paths[item]
        # 根据索引获取图片对应的标签
        label = self.labels[item]
        # 打开图片文件并将其转换为灰度图像
        image = Image.open(img_path).convert('L')
        # 如果存在图像转换操作，则对图像进行转换
        if self.transform:
            image = self.transform(image)
        # 将标签转换为 torch 张量，数据类型为 float32
        label = torch.tensor(label, dtype=torch.float32)
        # 返回处理后的图像和对应的标签
        return image, label


if __name__=="__main__":
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IT'
    dataset=N_IT_Extract(p_root_dir)
    print(len(dataset))

    # 确保 Jsons 目录存在
    json_dir = '../Jsons'
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    # 将路径和标签信息保存到 JSON 文件
    data = {
        "image_paths": dataset.image_paths,
        "labels": dataset.labels
    }
    json_path = os.path.join(json_dir, 'N_IT_Extract_paths.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)
