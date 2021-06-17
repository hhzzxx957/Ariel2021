"""Define and train the baseline model"""
import time
from functools import partial
import argparse
import numpy as np
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import *
from models import MLP, Baseline, DilatedNet
from utils import (ArielMLDataset, ChallengeMetric, Scheduler,
                   one_cycle, simple_transform)


def train(save_name, log_dir, device_id=0):
    torch.cuda.set_device(int(device_id))
    writer = SummaryWriter(log_dir=project_dir / f'outputs/{log_dir}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Training
    dataset_train = ArielMLDataset(lc_train_path,
                                   params_train_path,
                                   shuffle=True,
                                   start_ind=0,
                                   max_size=train_size,
                                   transform=simple_transform,
                                   device=device,
                                   seed=random_seed)
    # Validation
    dataset_val = ArielMLDataset(lc_train_path,
                                 params_train_path,
                                 shuffle=True,
                                 start_ind=train_size,
                                 max_size=val_size,
                                 transform=simple_transform,
                                 device=device,
                                 seed=random_seed)

    # Loaders
    loader_train = DataLoader(dataset_train,
                              batch_size=batch_size,
                              shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Define baseline model
    # model = Baseline(H1=H1, H2=H2).double().to(device)
    # model = MLP().double().to(device)
    model = DilatedNet().to(device)
    print(model)

    # Define Loss, metric and optimizer
    loss_function = L1Loss() #MSELoss() #L1Loss, ChallengeMetric()
    challenge_metric = ChallengeMetric()
    opt = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=50, gamma=0.25)
    # scheduler = Scheduler(opt, partial(one_cycle, t_max=epochs, pivot=0.3))

    # Lists to record train and val scores
    # train_losses = []
    # val_losses = []
    # val_scores = []
    best_val_score = 0.
    count = 0

    for epoch in range(1, 1 + epochs):
        print("epoch", epoch)
        train_loss = 0
        val_loss = 0
        val_score = 0
        model.train()
        for k, item in enumerate(loader_train):
            pred = model(item['lc'])
            loss = loss_function(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
        train_loss = train_loss / len(loader_train)
        model.eval()
        for k, item in enumerate(loader_val):
            pred = model(item['lc'])
            loss = loss_function(item['target'], pred)
            score = challenge_metric.score(item['target'], pred)
            val_loss += loss.detach().item()
            val_score += score.detach().item()
        val_loss /= len(loader_val)
        val_score /= len(loader_val)
        writer.add_scalar('valid loss', val_loss, epoch)
        writer.add_scalar('valid score', val_score, epoch)
        writer.add_scalar('train loss', train_loss, epoch)

        print('Training loss', round(train_loss, 6))
        print('Val loss', round(val_loss, 6))
        print('Val score', round(val_score, 2))
        print('learning rate', scheduler.get_last_lr())
        # train_losses += [train_loss]
        # val_losses += [val_loss]
        # val_scores += [val_score]

        # early stop
        if val_score > best_val_score:
            best_val_score = val_score
            if epoch >= save_from:
                torch.save(model,
                        project_dir / f'outputs/{save_name}/model_state.pt')
            count = 0
        else:
            count += 1
            if count >= 20:
                print('early stop, best epoch: ', epoch-count)
                break

        scheduler.step()
        # scheduler.step(epoch)
    writer.close()
    # np.savetxt(project_dir / f'outputs/{save_name}/train_losses.txt',
    #            np.array(train_losses))
    # np.savetxt(project_dir / f'outputs/{save_name}/val_losses.txt',
    #            np.array(val_losses))
    # np.savetxt(project_dir / f'outputs/{save_name}/val_scores.txt',
    #            np.array(val_scores))


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='MLP')
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    train(args.save_name, args.log_dir, device_id=args.device_id)
    print(
        f'Parameters: {args.save_name}, train_size: {train_size}, batch_size: {batch_size}'
    )
    print(f'Training time: {(time.time()-start_time)/60:.3f} mins')
