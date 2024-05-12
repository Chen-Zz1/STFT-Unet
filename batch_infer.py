from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tr
import os
import matplotlib.pyplot as plt

from model import FBPCONVNet
from utils import load_data, load_checkpoint, cmap_convert


# 确保你有正确地设置 device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 加载模型
model_dir = './checkpoints'
fbp_conv_net = FBPCONVNet().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(fbp_conv_net.parameters(), lr=0.0001)

fbp_conv_net, optimizer, epoch_start = load_checkpoint(fbp_conv_net, optimizer, model_dir)


# 定义图像转换
def transform(image):
    return tr.Compose([
        tr.Grayscale(num_output_channels=1),
        tr.ToTensor(),
        # tr.Normalize(mean=0.548, std=0.237)
    ])(image)

# 确保没有梯度计算
torch.set_grad_enabled(False)

# 定义图像目录
image_directory = './testimg/test/ipt'

# 定义保存预测结果的目录
output_directory = './testimg/test/out'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 获取所有图片文件名
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 按批处理图片
batch_size = 4  # 你可以根据需要和GPU内存大小调整这个值
for i in range(0, len(image_files), batch_size):
    batch_images = []
    filenames = []
    for filename in image_files[i:i+batch_size]:
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
        batch_images.append(transform(image).to(device))
        filenames.append(filename)

    # 将图片堆叠成一个批次
    batch_images = torch.stack(batch_images, dim=0)

    # 使用模型进行推理
    with torch.no_grad():
        outputs = fbp_conv_net(batch_images)

    # 处理每个批次的输出
    # 后续将推理出的图像保存
    for idx, output in enumerate(outputs):
        output_image = output.cpu().numpy()
        output_image = np.clip(output_image, 0, 1)  # 将像素值限制在 0 到 1 之间
        output_image = (output_image * 255).astype(np.uint8)  # 将像素值转换为 0 到 255 之间的整数
        output_image = output_image.squeeze()  # 去除批次维度

        # 创建 PIL 图像对象
        output_image_pil = Image.fromarray(output_image)

        # 构造输出图像路径
        output_filename = filenames[idx]
        output_path = os.path.join(output_directory, output_filename)

        # 保存图像
        output_image_pil.save(output_path)

print("所有图像的超分辨率已完成并保存。")
