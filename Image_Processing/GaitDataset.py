import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch

class P_GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # 构建 GAIT-IST 文件夹路径
        #path_class = os.path.join(root_dir, 'parkinsonian', '*', 'silhouettes')
        #构建GAIT-IT文件夹路径
        path_class=os.path.join((root_dir,'Parkinson','*','silhouettes','*'))
        silhouettes_folders = glob.glob(path_class)
        for silhouettes_folder in silhouettes_folders:
            # 遍历 silhouettes 目录下的子目录
            sub_folders = os.listdir(silhouettes_folder)
            for sub_folder in sub_folders:
                if sub_folder.endswith('back'):
                    sub_folder_path = os.path.join(silhouettes_folder, sub_folder)
                    for file in os.listdir(sub_folder_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(sub_folder_path, file)
                            self.image_paths.append(img_path)
                            self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        label = self.labels[item]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label



class N_GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历 DatasetB-1 和 DatasetB-2 文件夹
        for sub_folder in os.listdir(root_dir):
            sub_folder_path = os.path.join(root_dir, sub_folder)
            if os.path.isdir(sub_folder_path):
                # 查找 silhouettes 文件夹
                silhouettes_path = os.path.join(sub_folder_path, "silhouettes")
                if os.path.isdir(silhouettes_path):
                    for root, dirs, files in os.walk(silhouettes_path):
                        for dir in dirs:
                            sub_dir_path = os.path.join(root, dir)
                            for inner_root, inner_dirs, inner_files in os.walk(sub_dir_path):
                                nm_dirs = [d for d in inner_dirs if d.startswith('nm-')]
                                for nm_dir in nm_dirs:
                                    nm_dir_path = os.path.join(inner_root, nm_dir)
                                    target_dir = os.path.join(nm_dir_path, '090')
                                    if os.path.isdir(target_dir):
                                        for img_file in os.listdir(target_dir):
                                            img_path = os.path.join(target_dir, img_file)
                                            try:
                                                self.image_paths.append(img_path)
                                                self.labels.append(0)
                                            except:
                                                continue

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        label = self.labels[item]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label=torch.tensor(label,dtype=torch.float32)
        return image, label