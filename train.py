import os
import torchvision
import argparse
import numpy as np
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from tqdm import tqdm
import math

from model import CNN3D
from dataset import Senz3dDataset
from util import *
from i3dpt import *
import torch.nn.functional as F

train_path = "/home/dudupoo/data/senz3d_dataset/dataset/train/"
test_path = "/home/dudupoo/data/senz3d_dataset/dataset/test/"
img_x, img_y = 256, 256  # resize video 2d frame size
depth_x, depth_y = 320, 240
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 10, 1
learning_rate = 1e-4
_lambda = 0.05 # 50 x 10^-3


def train(args,
          model_rgb,
          model_depth,
          optimizer_rgb,
          optimizer_depth,
          train_loader,
          valid_loader,
          criterion,
          regularizer,
          corr,
          epoch):
    device = args.device
    model_rgb.train()
    model_depth.train()
    losses = []
    scores = []

    for batch_idx, (rgb, depth, y) in enumerate(train_loader):
        # distribute data to device
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        rgb_out, rgb_feature_map = model_rgb(rgb)
        depth_out, depth_feature_map = model_depth(depth)

        rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)
        depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
        print("RGB fmap transpose shape :: {}".format(rgb_feature_map_T.shape))
        print("depth fmap transpose shape :: {}".format(depth_feature_map_T.shape))

        rgb_corr = torch.mul(rgb_feature_map, rgb_feature_map_T)
        depth_corr = torch.mul(depth_feature_map, depth_feature_map_T)
        print("RGB correlation ::  {}".format(rgb_corr.shape))
        print("depth correlation :: {}".format(depth_corr.shape))

        loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
        loss_depth = criterion(depth_out, torch.max(y, 1)[1])

        print("RGB loss :: {}".format(loss_rgb))
        print("depth loss :: {}".format(loss_depth))

        focal_reg_param = regularizer(loss_rgb, loss_depth)

        print("focal regularizer parameter :: {} ".format(focal_reg_param))

        ssa_loss = focal_reg_param * (torch.sub(rgb_corr, depth_corr))

        print("ssa loss :: {} " .format(ssa_loss.shape))

        reg_loss_rgb = loss_rgb + (_lambda * ssa_loss)
        reg_loss_depth = loss_depth + (_lambda * ssa_loss)

        # print(reg_loss_rgb)
        # print(reg_loss_depth)

        reg_loss_rgb.backward(retain_graph=True)
        reg_loss_depth.backward()

        optimizer_rgb.step()
        optimizer_depth.step()

        # losses.append(loss_rgb.item())
        if epoch % 10 == 0 and batch_idx == 0:
            print("Regularizer : {}".format(focal_reg_param))
            print("Normal RGB loss : {} \t Regularized RGB loss : {} ".format(loss_rgb, reg_loss_rgb))
            print("Normal Depth loss : {} \t Regularized Depth loss : {} ".format(loss_depth, reg_loss_depth))


def main():
    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    train_videos_path = []
    test_videos_path = []

    for folder in os.listdir(train_path):
        train_videos_path.append(train_path + folder)

    for folder in os.listdir(test_path):
        test_videos_path.append(test_path + folder)

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    train_rgb_set = Senz3dDataset(train_videos_path, selected_frames, to_augment=False, mode='train')
    test_rgb_set = Senz3dDataset(test_videos_path, selected_frames, to_augment=False, mode='test')

    train_loader = data.DataLoader(train_rgb_set, pin_memory=True, batch_size=1)
    valid_loader = data.DataLoader(test_rgb_set, pin_memory=True, batch_size=1)

    # model_rgb_cnn = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y, num_classes=11).to(device)
    model_rgb_cnn = I3D(num_classes=11,
                 modality='rgb',
                 dropout_prob=0,
                 name='inception').to(device)
    # model_depth_cnn = CNN3D(t_dim=len(selected_frames), img_x=depth_x, img_y=depth_y, num_classes=11).to(device)
    model_depth_cnn = I3D(num_classes=11,
                        modality='rgb',
                        dropout_prob=0,
                        name='inception').to(device)
    optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
    optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
    criterion = torch.nn.CrossEntropyLoss()
    args = parse()
    args.device = device

    def regularizer(loss1, loss2):
        beta = 2.0
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0

    def get_correlation_matrix(feature_map):    # Unused
        variance, sample_mean = torch.var_mean(feature_map)
        # x = torchvision.transforms.Normalize(mean=sample_mean, std=variance)
        # normalized = x(feature_map)
        # print(normalized.shape)
        # sub_tensor = torch.sub(feature_map, sample_mean)
        # normalized_tensor = torch.addcdiv(sub_tensor, variance)
        # feature_map = feature_map.sub_(sample_mean).div_(variance)
        sample_mean = torch.as_tensor(sample_mean, dtype=feature_map.dtype, device=feature_map.device)
        variance = torch.as_tensor(variance, dtype=feature_map.dtype, device=feature_map.device)
        feature_map.sub_(sample_mean).div_(variance)
        return feature_map

    # print(model_rgb_cnn)


    for epoch in range(1):

        train(args=args,
              model_rgb=model_rgb_cnn,
              model_depth=model_depth_cnn,
              optimizer_rgb=optimizer_rgb,
              optimizer_depth=optimizer_depth,
              train_loader=train_loader,
              valid_loader=valid_loader,
              criterion=criterion,
              regularizer=regularizer,
              corr=get_correlation_matrix,
              epoch=epoch)


if __name__ == "__main__":
    main()
