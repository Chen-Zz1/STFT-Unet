from PIL import Image
import torchvision.transforms as tr
import random
from model import FBPCONVNet
import torch
import numpy as np
import math
import argparse
import os
from utils import load_data, load_checkpoint, cmap_convert

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义随机翻转函数
def random_flip(tag, ipt):
    rand = random.random()
    if rand < 0.25:
        return tr.functional.hflip(tag), tr.functional.hflip(ipt)
    elif rand < 0.5:
        return tr.functional.vflip(tag), tr.functional.vflip(ipt)
    elif rand < 0.75:
        return tr.functional.vflip(tr.functional.hflip(tag)), tr.functional.vflip(tr.functional.hflip(ipt))
    else:
        return tag, ipt

def transform(image):
    return tr.Compose([
        tr.Grayscale(num_output_channels=1),
        tr.ToTensor(),
        # tr.Normalize(mean=0.548, std=0.237)
    ])(image)

def data_argument(noisy, orig):

    # flip horizontal
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(2)
            orig[i] = orig[i].flip(2)

    # flip vertical
    for i in range(noisy.shape[0]):
        rate = np.random.random()
        if rate > 0.5:
            noisy[i] = noisy[i].flip(1)
            orig[i] = orig[i].flip(1)
    return noisy, orig


def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data loading
    print('load training data')
    image_directory = './testimg'
    tag_directory = image_directory + '/out/tag'
    ipt_directory = image_directory + '/out/ipt'
    image_files = [f for f in os.listdir(tag_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    epoch = config.epoch
    batch_size = config.batch_size
    grad_max = config.grad_max
    learning_rate = config.learning_rate

    smp_step = config.sample_step // batch_size * batch_size
    los_step = 50 // batch_size * batch_size


    fbp_conv_net = FBPCONVNet().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(fbp_conv_net.parameters(), lr=0.0001)
    epoch_start = 0

    # load check_point
    if os.path.exists(config.checkpoint_dir) and len(os.listdir(config.checkpoint_dir)) > 0:
        fbp_conv_net, optimizer, epoch_start = load_checkpoint(fbp_conv_net, optimizer, config.checkpoint_dir)

    fbp_conv_net.train()

    print('start training...')
    for e in range(epoch_start, epoch):

        # each epoch
        for i in range(0, len(image_files), batch_size):
            tag_batch_images = []
            ipt_batch_images = []
            for filename in image_files[i:i+batch_size]:
                tag_image_path = os.path.join(tag_directory, filename)
                ipt_image_path = os.path.join(ipt_directory, filename)
                tag_image = Image.open(tag_image_path)
                ipt_image = Image.open(ipt_image_path)

                tag_image, ipt_image = random_flip(tag_image, ipt_image)

                tag_batch_images.append(transform(tag_image).to(device))
                ipt_batch_images.append(transform(ipt_image).to(device))
            
            # 将batch图片堆叠成一个批次
            tag_batch_images = torch.stack(tag_batch_images, dim=0)
            ipt_batch_images = torch.stack(ipt_batch_images, dim=0)
            # data argument
            # noisy_batch, orig_batch = data_argument(noisy_batch, orig_batch)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward Propagation
            y_pred = fbp_conv_net(ipt_batch_images)


            # save sample images
            if i % smp_step == 0:
                if not os.path.exists(config.sample_dir):
                    os.mkdir(config.sample_dir)
                sample_img_path = os.path.join(config.sample_dir, 'epoch-%d-iteration-%d.jpg' % (e + 1, i + 1))
                sample_img = tr.ToPILImage()(y_pred[0].squeeze())
                sample_img.save(sample_img_path)
                print('save image:', sample_img_path)

            # Compute and print loss
            loss = criterion(y_pred, tag_batch_images)
            if i % los_step == 0:
                print('loss (epoch-%d-iteration-%d) : %f' % (e+1, i+1, loss.item()))

            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_value_(fbp_conv_net.parameters(), clip_value=grad_max)

            # Update the parameters
            optimizer.step()

        # shuffle data
        # ind = np.random.permutation(noisy.shape[0])
        # noisy = noisy[ind]
        # orig = orig[ind]

        # save check_point
        if (e+1) % config.checkpoint_save_step == 0 or (e+1) == config.epoch:
            if not os.path.exists(config.checkpoint_dir):
                os.mkdir(config.checkpoint_dir)
            check_point_path = os.path.join(config.checkpoint_dir, 'epoch-%d.pkl' % (e+1))
            torch.save({'epoch': e+1, 'state_dict': fbp_conv_net.state_dict(), 'optimizer': optimizer.state_dict()},
                       check_point_path)
            print('save checkpoint %s' % check_point_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=tuple, default=0.0001)
    parser.add_argument('--grad_max', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--data_path', type=str, default='./preproc_x20_ellipse_fullfbp.mat')
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--sample_dir', type=str, default='./samples/')
    parser.add_argument('--checkpoint_save_step', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    config = parser.parse_args()
    print(config)
    main(config)
