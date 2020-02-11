import os
import torchvision

import numpy as np
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from torch.optim import Adam
from tqdm import tqdm

from model import CNN3D
from dataset import Senz3dDataset

train_path = "/home/dudupoo/Downloads/senz3d_dataset/dataset/train/"
test_path = "/home/dudupoo/Downloads/senz3d_dataset/dataset/test/"
img_x, img_y = 256, 256  # resize video 2d frame size
depth_x, depth_y = 320, 240
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 30, 1
learning_rate = 1e-4


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

    train_loader = data.DataLoader(train_rgb_set, pin_memory=True)
    valid_loader = data.DataLoader(test_rgb_set, pin_memory=True)

    model_rgb_cnn = CNN3D(t_dim=len(selected_frames), img_x=img_x, img_y=img_y, num_classes=11).to(device)
    model_depth_cnn = CNN3D(t_dim=len(selected_frames), img_x=depth_x, img_y=depth_y, num_classes=11).to(device)
    optimizer_rgb = torch.optim.Adam(model_rgb_cnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
    optimizer_depth = torch.optim.Adam(model_depth_cnn.parameters(), lr=learning_rate)  # optimize all cnn parameters

    ############################################################################

    model_rgb_cnn.train()

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch
    for batch_idx, (X, z, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer_rgb.zero_grad()
        output = model_rgb_cnn(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        if batch_idx == 2:
            break
    ##############################################################################


if __name__ == "__main__":
    main()