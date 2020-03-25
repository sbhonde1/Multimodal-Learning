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
lr = 1e-4
_lambda = 0.05  # 50 x 10^-3


def train(args,
          model_rgb,
          model_depth,
          optimizer_rgb,
          optimizer_depth,
          train_loader,
          valid_loader,
          criterion,
          regularizer,
          epoch,
          tq):
    device = args.device
    model_rgb.train()
    model_depth.train()
    ssa_losses = []
    rgb_losses = []
    depth_losses = []
    rgb_regularized_losses = []
    depth_regularized_losses = []

    for batch_idx, (rgb, depth, y) in enumerate(train_loader):
        # distribute data to device
        rgb, depth, y = rgb.to(device), depth.to(device), y.to(device)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        rgb_out, rgb_feature_map = model_rgb(rgb)
        depth_out, depth_feature_map = model_depth(depth)

        rgb_feature_map_T = torch.transpose(rgb_feature_map, 1, 2)
        depth_feature_map_T = torch.transpose(depth_feature_map, 1, 2)
        # print("RGB fmap transpose shape :: {}".format(rgb_feature_map_T.shape))
        # print("depth fmap transpose shape :: {}".format(depth_feature_map_T.shape))

        rgb_corr = torch.mul(rgb_feature_map, rgb_feature_map_T)
        depth_corr = torch.mul(depth_feature_map, depth_feature_map_T)
        # print("RGB correlation ::  {}".format(rgb_corr.shape))
        # print("depth correlation :: {}".format(depth_corr.shape))

        loss_rgb = criterion(rgb_out, torch.max(y, 1)[1])  # index of the max log-probability
        loss_depth = criterion(depth_out, torch.max(y, 1)[1])

        # print("RGB loss :: {}".format(loss_rgb))
        # print("depth loss :: {}".format(loss_depth))

        focal_reg_param = regularizer(loss_rgb, loss_depth)

        # print("focal regularizer parameter :: {} ".format(focal_reg_param))

        ssa_loss = focal_reg_param * torch.sum((torch.sub(rgb_corr, depth_corr)) ** 2)

        # print("ssa loss :: {} " .format(ssa_loss.shape))

        reg_loss_rgb = loss_rgb + (_lambda * ssa_loss)
        reg_loss_depth = loss_depth + (_lambda * ssa_loss)

        # print(reg_loss_rgb)
        # print(reg_loss_depth)

        reg_loss_rgb.backward(retain_graph=True)
        reg_loss_depth.backward()

        optimizer_rgb.step()
        optimizer_depth.step()

        ssa_losses.append(ssa_loss.item())
        rgb_losses.append(loss_rgb.item())
        depth_losses.append(loss_depth.item())
        rgb_regularized_losses.append(reg_loss_rgb.item())
        depth_regularized_losses.append(reg_loss_depth.item())
        tq.update(1)
        # if epoch % 10 == 0 and batch_idx == 0:
        #     print("Regularizer : {}".format(focal_reg_param))
        #     print("Normal RGB loss : {} \t Regularized RGB loss : {} ".format(loss_rgb, reg_loss_rgb))
        #     print("Normal Depth loss : {} \t Regularized Depth loss : {} ".format(loss_depth, reg_loss_depth))
    mean_ssa = np.mean(ssa_losses)
    mean_rgb = np.mean(rgb_losses)
    mean_reg_rgb = np.mean(rgb_regularized_losses)
    mean_depth = np.mean(depth_losses)
    mean_reg_depth = np.mean(depth_regularized_losses)
    return mean_ssa, mean_rgb, mean_reg_rgb, mean_depth, mean_reg_depth


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
    optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=lr)  # optimize all cnn parameters
    criterion = torch.nn.CrossEntropyLoss()
    args = parse()
    args.device = device

    def regularizer(loss1, loss2):
        beta = 2.0
        if loss1 - loss2 > 0:
            return (beta * math.exp(loss1 - loss2)) - 1
        return 0.0

    # print(model_rgb_cnn)

    for epoch in range(1):
        tq = tqdm(total=(len(train_loader)))
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        ssa, rgb, reg_rgb, depth, reg_depth = train(args=args,
                                                    model_rgb=model_rgb_cnn,
                                                    model_depth=model_depth_cnn,
                                                    optimizer_rgb=optimizer_rgb,
                                                    optimizer_depth=optimizer_depth,
                                                    train_loader=train_loader,
                                                    valid_loader=valid_loader,
                                                    criterion=criterion,
                                                    regularizer=regularizer,
                                                    epoch=epoch,
                                                    tq=tq)
        tq.set_postfix(ssa_loss='{:.5f}'.format(ssa), regularized_rgb_loss='{:.5f}'.format(reg_rgb))

if __name__ == "__main__":
    main()
