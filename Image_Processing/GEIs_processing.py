import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import gc

# 处理单张图片的函数，从 Center_Processing.py 中复制过来
def process_image(img_file, img_size=224):
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
        return None

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
    return processed_img

# 合成能量图的函数
def generate_gei(images):
    if len(images) == 0:
        return None
    sum_img = np.sum(images, axis=0)
    gei = (sum_img / len(images)).astype('uint8')
    return gei

# 处理并保存能量图的函数
def process_and_save_geis(root_dir, output_folder, img_size=224):
    os.makedirs(output_folder, exist_ok=True)
    total_img_count = 0
    processed_img_count = 0

    # 计算总图片数
    for sub_folder in os.listdir(root_dir):
        sub_folder_path = os.path.join(root_dir, sub_folder)
        if os.path.isdir(sub_folder_path):
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
                                    total_img_count += len(os.listdir(target_dir))

    # 处理图片并生成能量图
    with tqdm(total=total_img_count, desc="Processing Images", unit="img") as pbar:
        for sub_folder in os.listdir(root_dir):
            sub_folder_path = os.path.join(root_dir, sub_folder)
            if os.path.isdir(sub_folder_path):
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
                                        processed_images = []
                                        for img_file in os.listdir(target_dir):
                                            img_path = os.path.join(target_dir, img_file)
                                            processed_img = process_image(img_path, img_size)
                                            if processed_img is not None:
                                                processed_images.append(processed_img)
                                            processed_img_count += 1
                                            pbar.update(1)
                                            pbar.set_postfix({"Processed/Total": f"{processed_img_count}/{total_img_count}"})
                                        if len(processed_images) > 0:
                                            gei = generate_gei(processed_images)
                                            if gei is not None:
                                                # 生成新的文件名格式
                                                parts = nm_dir_path.replace(silhouettes_path, "").strip(os.sep).split(os.sep)
                                                name_parts = "_".join(parts)
                                                gei_name = f'GEIs_back_{name_parts}.png'
                                                gei_path = os.path.join(output_folder, gei_name)
                                                Image.fromarray(gei).save(gei_path)
                                        # 清空 processed_images 列表
                                        del processed_images
                                        gc.collect()


if __name__ == "__main__":
    root_dir = r'D:\研究生课程\研究生课程\database\_BDataset'
    output_folder = r'D:\研究生课程\研究生课程\database\Gait_Dataset\normal\GEIs_back'
    process_and_save_geis(root_dir, output_folder, img_size=224)