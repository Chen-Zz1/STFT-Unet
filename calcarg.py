import os
import numpy as np
from PIL import Image

# 定义函数以加载图像文件夹中的所有灰度图像
def load_grayscale_images(folder_path):
    image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    grayscale_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # 'L'模式将图像转换为灰度图
        grayscale_images.append(img)
    return grayscale_images

# 定义函数以计算图像的平均值和标准差
def calculate_mean_and_std(images):
    images = np.array([np.array(img) for img in images], dtype=np.float32)
    # 将像素值映射到0到255范围内
    images = images / 255.0
    mean = np.mean(images)
    std = np.std(images)
    return mean, std

# 定义文件夹路径
folder_path = './testimg/out/tag'

# 加载灰度图像
grayscale_images = load_grayscale_images(folder_path)

# 计算图像的平均值和标准差
mean, std = calculate_mean_and_std(grayscale_images)
print("Mean:", mean)
print("Standard Deviation:", std)
