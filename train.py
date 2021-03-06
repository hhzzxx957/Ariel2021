"""Define and train the baseline model"""
import argparse
import time
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from constants import *
from models import Baseline, DilatedNet
from utils import (ArielMLDataset, ArielMLFeatDataset, ChallengeMetric, generate_indice,
                   Scheduler, one_cycle, simple_transform, subavg_transform)


def train(save_name, log_dir, device_id=0, draw_graph=False, valid=True):
    torch.cuda.set_device(int(device_id))
    writer = SummaryWriter(log_dir=project_dir / f'outputs/{log_dir}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    indices_tot = list(range(1256))
    np.random.seed(random_seed)
    np.random.shuffle(indices_tot)
    # train_ind = indices_tot[:train_size]
    # valid_ind = indices_tot[train_size:train_size+val_size]

    if valid:
        train_ind = indices_tot[:int(train_size*0.9)]
        train_ind = generate_indice(train_ind)
    else:
        train_ind = generate_indice(indices_tot)

    # Training
    dataset_train = ArielMLFeatDataset(data_train_path,
                                   feat_train_path,
                                   lgb_train_path,
                                #    filefeat_train_path,
                                   quantilefeat_train_path,
                                #    quantilephotonfeat_train_path,
                                   sample_ind=train_ind,
                                   transform=subavg_transform, #simple_transform,
                                   device=device)
    # Loaders
    loader_train = DataLoader(dataset_train,
                              batch_size=batch_size,
                              shuffle=True,)
                  
    if valid:
        valid_ind = indices_tot[int(train_size*0.9):int(train_size)]
        valid_ind = generate_indice(valid_ind)
        # Validation
        dataset_val = ArielMLFeatDataset(data_train_path,
                                    feat_train_path,
                                    lgb_train_path,
                                    # filefeat_train_path,
                                    quantilefeat_train_path,
                                    # quantilephotonfeat_train_path,
                                    sample_ind=valid_ind,
                                    transform=subavg_transform, #simple_transform,
                                    device=device)

        loader_val = DataLoader(dataset_val, batch_size=batch_size)

    # Define model
    # model = Baseline(H1=H1, H2=H2).double().to(device)
    # model = MLP().double().to(device)
    model = DilatedNet(add_feat=True,).to(device) # add_fft=True
    # model = DilatedCNNLSTMNet(add_feat=True).to(device)
    print(model)

    if draw_graph:
        item= iter(loader_val).next()
        writer.add_graph(model, input_to_model=(item['lc'],item['feat']),verbose=True)
        writer.close()
        return 0

    # Define Loss, metric and optimizer
    loss_function = L1Loss()  #MSELoss() #L1Loss, ChallengeMetric()
    challenge_metric = ChallengeMetric()
    opt = AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(opt, step_size=50, gamma=0.25)
    # scheduler = Scheduler(opt, partial(one_cycle, t_max=epochs, pivot=0.3))

    best_val_score = 0.
    count = 0

    for epoch in range(1, 1 + epochs):
        print("epoch", epoch)
        train_loss = 0
        val_loss = 0
        val_score = 0
        model.train()
        for k, item in enumerate(loader_train):
            pred = model(item['lc'], item['feat'])
            loss = loss_function(item['target'], pred)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
        train_loss = train_loss / len(loader_train)
        writer.add_scalar('train loss', train_loss, epoch)

        print('learning rate', scheduler.get_last_lr())
        print('Training loss', round(train_loss, 6))

        if valid:
            model.eval()
            for k, item in enumerate(loader_val):
                pred = model(item['lc'], item['feat'])
                loss = loss_function(item['target'], pred)
                score = challenge_metric.score(item['target'], pred)
                val_loss += loss.detach().item()
                val_score += score.detach().item()
            val_loss /= len(loader_val)
            val_score /= len(loader_val)
            writer.add_scalar('valid loss', val_loss, epoch)
            writer.add_scalar('valid score', val_score, epoch)

            print('Val loss', round(val_loss, 6))
            print('Val score', round(val_score, 2))
        

        # early stop
        if valid:
            if val_score > best_val_score:
                best_val_score = val_score
                if epoch >= save_from:
                    torch.save(model,
                            project_dir / f'outputs/{save_name}/model_state.pt')
        else:
            if epoch >= save_from:
                torch.save(model,
                        project_dir / f'outputs/{save_name}/model_state.pt')
        scheduler.step()
        # scheduler.step(epoch)
    writer.close()


def cross_valid_train(save_name, log_dir, device_id=0, nsplit=10):
    torch.cuda.set_device(int(device_id))
    writer = SummaryWriter(log_dir=project_dir / f'outputs/{log_dir}')

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    indices_tot = list(range(1256))
    kf = KFold(n_splits=nsplit, shuffle=True, random_state=random_seed)
    for i, (train_ind, valid_ind) in enumerate(kf.split(indices_tot)):
        # Training
        train_ind = generate_indice(train_ind)
        valid_ind = generate_indice(valid_ind)
        dataset_train = ArielMLFeatDataset(data_train_path,
                                    feat_train_path,
                                    lgb_train_path,
                                    # filefeat_train_path,
                                    quantilefeat_train_path,
                                    # quantilephotonfeat_train_path,
                                    sample_ind=train_ind,
                                    transform=subavg_transform, #simple_transform,
                                    device=device)
        # Validation
        dataset_val = ArielMLFeatDataset(data_train_path,
                                    feat_train_path,
                                    lgb_train_path,
                                    # filefeat_train_path,
                                    quantilefeat_train_path,
                                    # quantilephotonfeat_train_path,
                                    sample_ind=valid_ind,
                                    transform=subavg_transform, #simple_transform,
                                    device=device)

        # Loaders
        loader_train = DataLoader(dataset_train,
                                batch_size=batch_size,
                                shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size=batch_size)

        # Define model
        # DilatedCNNLSTMNet, DilatedFeatNet
        # model = Baseline(H1=H1, H2=H2).double().to(device)
        # model = MLP().double().to(device)
        model = DilatedNet(add_feat=True).to(device)
        # model = DilatedCNNLSTMNet(add_feat=True).to(device)
        print(model)

        # Define Loss, metric and optimizer
        loss_function = L1Loss()  #MSELoss() #L1Loss, ChallengeMetric()
        challenge_metric = ChallengeMetric()
        opt = AdamW(model.parameters(), lr=lr)
        scheduler = StepLR(opt, step_size=50, gamma=0.25)
        # scheduler = Scheduler(opt, partial(one_cycle, t_max=epochs, pivot=0.3))

        best_val_score = 0.
        count = 0

        for epoch in range(1, 1 + epochs):
            print("epoch", epoch)
            train_loss = 0
            val_loss = 0
            val_score = 0
            model.train()
            for k, item in enumerate(loader_train):
                pred = model(item['lc'], item['feat'])
                loss = loss_function(item['target'], pred)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.detach().item()
            train_loss = train_loss / len(loader_train)
            model.eval()
            for k, item in enumerate(loader_val):
                pred = model(item['lc'],item['feat'])
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

            # early stop
            if val_score > best_val_score:
                best_val_score = val_score
                if epoch >= save_from:
                    torch.save(model,
                            project_dir / f'outputs/{save_name}/model_state_{i}.pt')

            scheduler.step()
            # scheduler.step(epoch)
    writer.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='MLP')
    parser.add_argument('--log_dir', type=str, default='test')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    cross_valid_train(args.save_name, args.log_dir, device_id=args.device_id)
    print(
        f'Parameters: {args.save_name}, train_size: {train_size}, batch_size: {batch_size}'
    )
    print(f'Training time: {(time.time()-start_time)/60:.3f} mins \n')
