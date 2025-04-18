import json

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import glob
from tqdm import tqdm

class P_IST_GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None, sub_folder_type='silhouettes', view_type='_front'):
        # 存储数据集的根目录
        self.root_dir = root_dir
        # 存储图像预处理的转换操作
        self.transform = transform
        # 存储所有图像的文件路径
        self.image_paths = []
        # 存储每个图像对应的标签
        self.labels = []
        # 存储用户选择的子文件夹类型（'silhouettes' 或 'GEIs'）
        self.sub_folder_type = sub_folder_type
        # 存储用户选择的视图类型（'_front' 或 '_back'）
        self.view_type = view_type

        # 构建 GAIT - IST 文件夹路径，使用用户选择的子文件夹类型
        path_class = os.path.join(root_dir, 'P_IST_GEIs', '*', self.sub_folder_type)
        # 使用 glob 模块获取所有符合条件的文件夹路径
        folders = glob.glob(path_class)

        # 遍历所有符合条件的文件夹
        for folder in folders:
            # 获取当前文件夹下的所有子文件夹
            sub_folders = os.listdir(folder)
            # 遍历每个子文件夹
            for sub_folder in sub_folders:
                # 检查子文件夹名称是否以用户选择的视图类型结尾
                if sub_folder.lower().endswith(self.view_type):
                    # 构建子文件夹的完整路径
                    sub_folder_path = os.path.join(folder, sub_folder)
                    # 遍历子文件夹下的所有文件
                    for file in os.listdir(sub_folder_path):
                        # 检查文件是否为图片文件（支持 .png, .jpg, .jpeg 格式）
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # 构建图片文件的完整路径
                            img_path = os.path.join(sub_folder_path, file)
                            # 将图片文件路径添加到 image_paths 列表中
                            self.image_paths.append(img_path)
                            # 为该图片添加标签 1（表示 parkinsonian 患者）
                            self.labels.append(1)

    def __len__(self):
        # 返回数据集的长度，即图像的数量
        return len(self.image_paths)

    def __getitem__(self, item):
        # 根据索引获取图像的文件路径
        img_path = self.image_paths[item]
        # 根据索引获取图像对应的标签
        label = self.labels[item]
        # 打开图像文件并将其转换为灰度图像
        image = Image.open(img_path).convert('L')
        # 如果存在预处理转换操作，则对图像进行转换
        if self.transform:
            image = self.transform(image)
        # 将标签转换为 torch.Tensor 类型，数据类型为 torch.float32
        label = torch.tensor(label, dtype=torch.float32)
        # 返回处理后的图像和对应的标签
        return image, label

if __name__=="__main__":
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IST'
    dataset=P_IST_GaitDataset(p_root_dir)
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
    json_path = os.path.join(json_dir, 'P_IST_Extract_paths_byour_front.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)