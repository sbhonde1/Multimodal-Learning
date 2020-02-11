import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import torchvision.transforms as transforms

img_x, img_y = 256, 256  # resize video 2d frame size

transform_rgb = transforms.Compose([transforms.Resize([img_x, img_y]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

transform_depth = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5], std=[0.5])])


class Senz3dDataset(Dataset):
    def __init__(self, folders, frames, to_augment=False, transform=None, mode='train'):
        """
        :param folders: list of all the video folders
        :param frames: start frame, end frame and skip frame numpy array
        :param to_augment: boolean if the data is suppose to augment
        :param transform: transform function
        :param mode: train/test
        """
        self.folders = folders
        self.frames = frames
        self.to_augment = to_augment
        self.mode = mode
        self.transform = transform
        # convert labels -> category
        self.le = LabelEncoder()
        action_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.le.fit(action_names)
        # convert category -> 1-hot
        self.action_category = self.le.transform(action_names).reshape(-1, 1)
        self.enc = OneHotEncoder()
        self.enc.fit(self.action_category)

    def __len__(self):
        return len(self.folders)

    def read_images(self, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(selected_folder, '{}-color.png'.format(i))).convert('L')
            image = transform_rgb(image)
            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)
        return X

    def read_depth(self, selected_folder, use_transform):
        X = []
        for i in self.frames:
            depth = np.fromfile(os.path.join(selected_folder, '{}-depth.bin'.format(i)), dtype='int16')
            depth = depth.reshape([240, 320])
            depth = Image.fromarray(depth, mode="L")
            depth = transform_depth(depth)
            X.append(depth.squeeze_(0))

        X = torch.stack(X, dim=0)
        return X

    def read_label(self, path):
        with open(path) as file:
            return int(file.read())

    def __getitem__(self, idx):
        # mode options is given if there's a need to experiment differently on train and valid data
        folder_name = self.folders[idx]
        # Load data
        rgb = self.read_images(folder_name, self.transform).unsqueeze_(0)  # (rgb)
        print(rgb.shape)
        depth = self.read_depth(folder_name, self.transform).unsqueeze_(0) # (depth)
        print(depth.shape)
        label = self.read_label(os.path.join(folder_name, 'label.txt'))
        y = torch.LongTensor(self.enc.transform([[label]]).toarray())
        return rgb, depth, y
