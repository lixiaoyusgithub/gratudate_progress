import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm

# 合成能量图的函数
def generate_gei(images):
    if len(images) == 0:
        return None
    sum_img = np.sum(images, axis=0)
    gei = (sum_img / len(images)).astype('uint8')
    return gei

class P_IT_GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None, data_type='silhouettes', new_folder_name='P_IT_GEIs'):
        # 存储数据集的根目录路径，后续将基于此路径查找数据文件
        self.root_dir = root_dir
        # 存储图像转换操作，例如数据增强、归一化等，默认为 None
        self.transform = transform
        # 用于存储图像文件的路径，方便后续根据索引获取图像
        self.image_paths = []
        # 用于存储每个图像对应的标签，这里标签统一为 1
        self.labels = []
        # 存储要使用的数据类型，默认为 'GEIs'
        self.data_type = data_type
        # 存储能量图的输出根文件夹，使用新的文件夹名称
        self.output_root = os.path.join(root_dir, new_folder_name)

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
                # 构建子目录的完整路径
                sub_folder_path = os.path.join(folder, sub_folder)
                # 获取子文件夹下的所有图片文件
                img_files = [os.path.join(sub_folder_path, file) for file in os.listdir(sub_folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # 构建对应的输出文件夹
                relative_path = os.path.relpath(sub_folder_path, os.path.join(root_dir, 'Normal'))
                output_folder = os.path.join(self.output_root, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                self.generate_geis_from_images(img_files, sub_folder, output_folder)

    def generate_geis_from_images(self, img_files, sub_folder, output_folder):
        total_img_count = len(img_files)
        # 使用 tqdm 显示进度条
        with tqdm(total=total_img_count, desc=f"Processing {sub_folder}", unit="img") as pbar:
            for i in range(0, total_img_count):
                if i + 30 > total_img_count:
                    break
                processed_images = []
                for j in range(30):
                    img_path = img_files[i + j]
                    try:
                        img = Image.open(img_path).convert('L')
                        img = np.array(img)
                        processed_images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img_path}: {e}")
                    pbar.update(1)
                if len(processed_images) > 0:
                    gei = generate_gei(processed_images)
                    if gei is not None:
                        # 生成新的文件名格式
                        gei_name = f'GEIs_{sub_folder}_{i // 30}.png'
                        gei_path = os.path.join(output_folder, gei_name)
                        Image.fromarray(gei).save(gei_path)
                        self.image_paths.append(gei_path)
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


if __name__ == "__main__":
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IT'
    dataset = P_IT_GaitDataset(p_root_dir)