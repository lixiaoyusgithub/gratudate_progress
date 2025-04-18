import glob
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm

# 合成能量图的函数
def generate_gei(images):
    if len(images) == 0:
        return None
    sum_img = np.sum(images, axis=0)
    gei = (sum_img / len(images)).astype('uint8')
    return gei

class N_IST_Generate_GEIs(Dataset):
    def __init__(self, root_dir, transform=None, sub_folder_type='silhouettes', new_folder_name='generated_geis'):
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
        # 存储能量图的输出根文件夹，使用新的文件夹名称
        self.output_root = os.path.join(root_dir, new_folder_name)

        # 构建 GAIT - IST 文件夹路径，使用用户选择的子文件夹类型
        path_class = os.path.join(root_dir, 'normal', '*', self.sub_folder_type)
        # 使用 glob 模块获取所有符合条件的文件夹路径
        folders = glob.glob(path_class)

        # 遍历所有符合条件的文件夹
        for folder in folders:
            # 获取当前文件夹下的所有子文件夹
            sub_folders = os.listdir(folder)
            # 遍历每个子文件夹
            for sub_folder in sub_folders:
                # 构建子文件夹的完整路径
                sub_folder_path = os.path.join(folder, sub_folder)
                # 获取子文件夹下的所有图片文件
                img_files = [os.path.join(sub_folder_path, file) for file in os.listdir(sub_folder_path) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
                # 构建对应的输出文件夹
                relative_path = os.path.relpath(sub_folder_path, os.path.join(root_dir, 'normal'))
                output_folder = os.path.join(self.output_root, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                self.generate_geis_from_images(img_files, sub_folder, output_folder)

    def generate_geis_from_images(self, img_files, sub_folder, output_folder):
        total_img_count = len(img_files)
        # 使用 tqdm 显示进度条
        with tqdm(total=total_img_count, desc=f"Processing {sub_folder}", unit="img") as pbar:
            for i in range(0, total_img_count):
                if i + 10 > total_img_count:
                    break
                processed_images = []
                for j in range(10):
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
                        gei_name = f'GEIs_{sub_folder}_{i // 10}.png'
                        gei_path = os.path.join(output_folder, gei_name)
                        Image.fromarray(gei).save(gei_path)

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

if __name__ == "__main__":
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IST'
    dataset = N_IST_Generate_GEIs(p_root_dir, new_folder_name='N_IST_GEIs')