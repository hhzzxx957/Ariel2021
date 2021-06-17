import argparse
import datetime
import gc
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from constants import *
from utils import ArielMLDataset, ChallengeMetric, simple_transform


def prediction(model_dir=None, save_name='MLP', device_id=0):
    torch.cuda.set_device(device_id)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    dataset_test = ArielMLDataset(lc_train_path,
                                  params_train_path,
                                  start_ind=train_size + val_size,
                                  max_size=test_size,
                                  transform=simple_transform)
    dataset_eval = ArielMLDataset(lc_test_path,
                                  shuffle=False,
                                  transform=simple_transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=1000, shuffle=False)

    if model_dir is None:
        model_dir = f'outputs/{save_name}/model_state.pt'
    model = torch.load(model_dir, map_location=device)
    challenge_metric = ChallengeMetric()

    # Test
    naive_1 = lambda x: torch.ones(x.shape[:-1]) * 0.06

    item = next(iter(loader_test))
    item['lc'] = item['lc'].to(device)
    preds = {
        'naive1': naive_1(item['lc']),
        'normal_1000ppm': torch.normal(item['target'], 1e-3),
        'model': model(item['lc'])
    }

    for name, pred in preds.items():
        print(
            name,
            f"\t{challenge_metric.score(item['target'], pred.cpu()).item():.2f}"
        )

    # Evaluation
    preds = []
    print('Evaluate length', len(loader_eval))
    for k, item in tqdm(enumerate(loader_eval)):
        gc.collect()
        item['lc'] = item['lc'].to(device)
        preds += [model(item['lc']).detach().cpu().numpy()]

    # eval_pred = torch.cat(preds)
    eval_pred = np.concatenate(preds, axis=0)
    print(eval_pred.shape)

    save_path = f'outputs/{save_name}/evaluation_{datetime.datetime.today().date()}.txt'
    if save_path and (53900, 55) == eval_pred.shape:
        np.savetxt(save_path, eval_pred, fmt='%.10f', delimiter='\t')


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='MLP')
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()
    prediction(save_name=args.save_name, device_id=args.device_id)
    print(f'Inference time: {(time.time()-start_time)/60:.3f} mins')
