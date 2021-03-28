import argparse
import os
import random

import sklearn.metrics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import configs as config
import utils


def train(gpu, args):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.gpu_nums, rank=gpu)

    net, start_epoch = utils.get_model(args.backbone,
                                       args.load_weights,
                                       args.model_dir,
                                       gpu)

    torch.cuda.set_device(gpu)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=args.patience, eps=1e-08)

    train_dataset = utils.get_dataset(args.data_dir, 'train')
    val_dataset = utils.get_dataset(args.data_dir, 'val')

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.gpu_nums, rank=gpu)
    data_loader_train = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_workers // args.gpu_nums,
                                                    pin_memory=True,
                                                    sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=args.gpu_nums, rank=gpu)
    data_loader_val = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=args.num_workers // args.gpu_nums,
                                                  pin_memory=True,
                                                  sampler=val_sampler)

    for epoch in range(start_epoch, args.num_epochs):

        running_loss = 0.0
        val_loss = 0.0
        train_iter = iter(data_loader_train)
        pbar = range(len(data_loader_train))
        if gpu == 0:
            pbar = tqdm(pbar)

        for i in pbar:
            x, y = train_iter.next()

            x = x.cuda(non_blocking=True)

            pred = net(x)

            loss = utils.calc_closs(pred, y.cuda(non_blocking=True))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            train_loss = running_loss / (i + 1)
            if gpu == 0:
                pbar.set_description('[%d/%d] Loss: %f' % (epoch, args.num_epochs, train_loss))

            if i == len(data_loader_train) - 1:

                if gpu == 0:
                    utils.save_model(args.model_dir, net.state_dict(), epoch)
                net.eval()

                with torch.no_grad():

                    for val_x, val_y in data_loader_val:
                        val_x = val_x.cuda(non_blocking=True)
                        val_pred = net(val_x)

                        loss = utils.calc_closs(val_pred, val_y.cuda(non_blocking=True))
                        val_loss += loss.item()

                val_loss = val_loss / len(data_loader_val)
                val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
                dist.all_reduce(val_loss)
                val_loss = val_loss.item() / args.gpu_nums

                if gpu == 0:
                    res_of_epoch = '[%d/%d] Train: %f; Val: %f; lr: %f;' % (
                                       epoch, args.num_epochs, train_loss, val_loss, optimizer.param_groups[0]['lr'])

                    pbar.set_description(res_of_epoch)

                net.train()

        scheduler.step(val_loss)
        train_iter = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--load_weights', type=int, default=None, help='number of epoch to load weights from')
    parser.add_argument('--backbone', type=str, default=config.backbone)
    parser.add_argument('--gpu_nums', type=int, default=torch.cuda.device_count(), help='number of gpus')
    parser.add_argument('--patience', type=int, default=3, help='patience for scheduler')

    parser.add_argument('--model_dir', type=str, default='/workspace/weights', help='dir to store model weights')
    parser.add_argument('--data_dir', type=str, default='/workspace/data', help='dir of data to train')

    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    random.seed(42)
    torch.manual_seed(42)
    mp.spawn(train, nprocs=args.gpu_nums, args=(args,))
