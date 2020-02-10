import os
import torchvision.transforms as transforms
import numpy as np
from dataset import Senz3dDataset
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.optim import Adam
from tqdm import tqdm


train_path = "/home/dudupoo/Downloads/senz3d_dataset/dataset/train/"
test_path = "/home/dudupoo/Downloads/senz3d_dataset/dataset/test/"
img_x, img_y = 256, 342  # resize video 2d frame size
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 30, 1


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

    # image transformation
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    train_set = Senz3dDataset(train_videos_path, selected_frames, to_augment=False, transform=transform, mode='train')
    test_set = Senz3dDataset(test_videos_path, selected_frames, to_augment=False, transform=transform, mode='test')

    train_loader = data.DataLoader(train_set, pin_memory=True)
    valid_loader = data.DataLoader(test_set, pin_memory=True)



if __name__ == "__main__":
    main()