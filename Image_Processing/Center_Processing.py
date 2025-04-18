import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import glob
from tqdm import tqdm

class P_GaitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        '''
        # 构建 GAIT-IST 文件夹路径
        path_class = os.path.join(root_dir, 'parkinsonian', '*', 'silhouettes')
        silhouettes_folders = glob.glob(path_class)
        for silhouettes_folder in silhouettes_folders:
            # 遍历 silhouettes 目录下的子目录
            sub_folders = os.listdir(silhouettes_folder)
            for sub_folder in sub_folders:
                if sub_folder.lower().endswith('_front'):
                    sub_folder_path = os.path.join(silhouettes_folder, sub_folder)
                    for file in os.listdir(sub_folder_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(sub_folder_path, file)
                            self.image_paths.append(img_path)
                            self.labels.append(1)
                else:
                    print(f"Skipping folder: {sub_folder}")  # 添加调试信息
        '''

        #构建 GAIT-IT目录读取
        path_class_it = os.path.join(root_dir, 'Parkinson', '*', 'GEIs', '*')
        silhouettes_folders_it = glob.glob(path_class_it)
        for silhouettes_folder in silhouettes_folders_it:
            # 遍历 silhouettes 目录下的子目录
            sub_folders = os.listdir(silhouettes_folder)
            for sub_folder in sub_folders:
                if sub_folder.endswith('_front'):
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
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def process_image(img_file, output_folder, img_size=224):
    # 使用 PIL.Image 读取图片并转换为灰度图
    img = Image.open(str(img_file)).convert('L')
    img = np.array(img)

    # Get the upper and lower points
    y_sum = img.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    img = img[y_top: y_btm + 1, :]

    ratio = img.shape[1] / img.shape[0]
    img = np.array(Image.fromarray(img).resize((int(img_size * ratio), img_size), Image.BICUBIC))

    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = img.sum(axis=0).cumsum()
    x_center = None
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break

    if not x_center:
        return

    # Get the left and right points
    half_width = img_size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        _ = np.zeros((img.shape[0], half_width))
        img = np.concatenate([_, img, _], axis=1)

    processed_img = img[:, left: right].astype('uint8')

    # Save processed images as PNG
    file_name = os.path.basename(img_file)
    base_name, extension = os.path.splitext(file_name)
    new_name = f'GEIs_front_' + base_name + '.png'
    new_path = os.path.join(output_folder, new_name)
    counter = 1
    while os.path.exists(new_path):
        new_name = f'GEIs_front_{base_name}_{counter}{extension}'
        new_path = os.path.join(output_folder, new_name)
        counter += 1

    Image.fromarray(processed_img).save(new_path)


def process_and_save_images(root_dir, output_folder, class_name, img_size=224):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if class_name == 'p':
        dataset = P_GaitDataset(root_dir, transform=transform)
    else:
        dataset = N_GaitDataset(root_dir, transform=transform)

    os.makedirs(output_folder, exist_ok=True)
    # 使用 tqdm 显示进度条
    with tqdm(total=len(dataset.image_paths), desc="Processing Images", unit="img") as pbar:
        for img_path in dataset.image_paths:
            process_image(img_path, output_folder, img_size)
            pbar.update(1)


if __name__ == "__main__":
    #p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IST'
    p_root_dir = r'D:\研究生课程\研究生课程\database\GAIT-IT'
    #n_root_dir = r'D:\研究生课程\研究生课程\database\_BDataset'
    p_output_folder = r'D:\研究生课程\研究生课程\database\Gait_Dataset\parkinsonian\GEIs_front_it'
    #n_output_folder = r'D:\研究生课程\研究生课程\database\Gait_Dataset\normal\silouettes_back'

    process_and_save_images(p_root_dir, p_output_folder, 'p', img_size=224)
    #process_and_save_images(n_root_dir, n_output_folder, 'n', img_size=224)