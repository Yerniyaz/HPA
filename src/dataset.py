import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset

import configs as config


def preprocess(img, transform=None):
    h, w, _ = img.shape
    left = 0
    right = 0
    top = 0
    bottom = 0
    if w > h:
        h = int(h * config.img_w / w)
        w = config.img_w
        top = (config.img_h - h) // 2
        bottom = config.img_h - h - top
    else:
        w = int(w * config.img_h / h)
        h = config.img_h
        left = (config.img_w - w) // 2
        right = config.img_w - w - left
    x = cv2.resize(img, (w, h))

    if transform is not None:
        x = transform(image=x)['image']

    x = cv2.copyMakeBorder(x, top, bottom, left, right,
                           cv2.BORDER_CONSTANT, value=(0, 0, 0))

    x = x.astype(np.float32) / 255.
    x = x.transpose(2, 1, 0)
    x = torch.tensor(x)

    return x


class CellDataset(Dataset):
    def __init__(self, data_dir, csv_files, transform=None, train=True):
        super().__init__()

        self.data_dir = data_dir
        csv_files = [os.path.join(data_dir, csv_file) for csv_file in csv_files]
        self.transform = transform

        if len(csv_files) == 1:
            chunks = pd.read_csv(csv_files[0], chunksize=100000)
            df = pd.concat(chunks)
        else:
            df = pd.concat([pd.read_csv(csv_file) for csv_file in csv_files])

        if train:
            df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        self.cell_types = df[config.cell_types].values
        self.img_ids = df['ID'].values
        self.train = train

    def __len__(self):
        return len(self.img_ids)

    def get_image(self, index):
        _id = self.img_ids[index % self.__len__()]
        red_path = os.path.join(self.data_dir, 'train_images', _id + '_red.png')
        green_path = red_path.replace('_red.png', '_green.png')
        blue_path = red_path.replace('_red.png', '_blue.png')
        yellow_path = red_path.replace('_red.png', '_yellow.png')

        img_red = cv2.imread(red_path, 0)
        img_green = cv2.imread(green_path, 0)
        img_blue = cv2.imread(blue_path, 0)
        img_yellow = cv2.imread(yellow_path, 0)

        if img_red is None or img_blue is None or img_green is None or img_yellow is None:
            return None

        x = np.stack([img_red, img_green, img_blue, img_yellow]).transpose(1, 2, 0)

        return x

    def preprocess_x(self, x):
        if self.train:
            x = preprocess(x, transform=self.transform)
        else:
            x = preprocess(x)
        return x

    def __getitem__(self, index):
        x = self.get_image(index)
        if x is None:
            return self[index + 1]

        x = self.preprocess_x(x)
        y = self.cell_types[index]
        y = torch.from_numpy(y).float()

        return x, y
