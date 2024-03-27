import os
from PIL import Image
import torchvision.transforms as transforms

# 定义转换
transform_tag = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),  # 将图像中心裁剪为512x512
    transforms.Grayscale(),
#     transforms.Resize(128),      # 将图像大小调整为128x128
    transforms.ToTensor()        # 将图像转换为张量
])

transform_ipt = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),  # 将图像中心裁剪为512x512
    transforms.Grayscale(),
    transforms.Resize(128),      # 将图像大小调整为128x128
    transforms.ToTensor()        # 将图像转换为张量
])
# 图像文件夹路径
folder_path = './testimg/src'

# 裁剪和调整大小的图像将保存在的文件夹路径
output_ipt_folder = './testimg/out/ipt'

# 标签图像的保存路径
output_tag_folder = './testimg/out/tag'

# 确保输出文件夹存在
os.makedirs(output_ipt_folder, exist_ok=True)
os.makedirs(output_tag_folder, exist_ok=True)

# 加载和处理图像
for filename in os.listdir(folder_path):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.JPEG')):  # 确保是图像文件
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        output_tag_image = transform_tag(image)
        
        # 保存处理后的图像
        output_tag_path = os.path.join(output_tag_folder, filename)
        output_image = transforms.ToPILImage()(output_tag_image)  # 转换为PIL图像以保存
        output_image.save(output_tag_path)

        output_ipt_image = transform_ipt(image)
        
        # 保存处理后的图像
        output_ipt_path = os.path.join(output_ipt_folder, filename)
        output_image = transforms.ToPILImage()(output_ipt_image)  # 转换为PIL图像以保存
        output_image.save(output_ipt_path)

print("图片处理完成！")
