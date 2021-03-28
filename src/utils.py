import os

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import configs as config
from dataset import CellDataset
from base_model import Net



def get_dataset(data_dir, type_key):
    params = config.params[type_key]
    params['data_dir'] = data_dir
    dataset = CellDataset(**params)

    return dataset


def get_model(backbone, load_weights, model_dir, gpu):
    net = Net(backbone, num_classes=len(config.cell_types)).cuda(gpu)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    start_epoch = 0
    if load_weights is not None:
        net.load_state_dict(torch.load(os.path.join(model_dir, f'epoch_{load_weights}' + config.checkpoint_ext)))
        start_epoch = load_weights + 1

    print(f'Loading from the checkpoint with epoch number {start_epoch}')
    return net, start_epoch


def save_model(model_dir, model, epoch):
    torch.save(model, os.path.join(model_dir, f'epoch_{epoch}' + config.checkpoint_ext))
    # logger.info(f'Saving checkpoint model with epoch number {epoch}')


def calc_closs(pred, y):
    loss_model = F.binary_cross_entropy_with_logits(pred, y)

    return loss_model

