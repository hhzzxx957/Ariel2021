import argparse
import datetime
import gc
import time

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from constants import *
from utils import ArielMLDataset, ChallengeMetric, simple_transform, ArielMLFeatDataset, subavg_transform


def prediction(model_dir=None, save_name='MLP', device_id=0):
    torch.cuda.set_device(device_id)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    indices_tot = list(range(125600))
    np.random.seed(random_seed)
    np.random.shuffle(indices_tot)
    valid_ind = indices_tot[train_size:train_size+test_size]
    test_ind = list(range(53900))

    dataset_test = ArielMLFeatDataset(data_train_path,
                                  feat_train_path,
                                  lgb_train_path,
                                  sample_ind=valid_ind,
                                  transform=subavg_transform, #simple_transform,
                                  device=device)
    dataset_eval = ArielMLFeatDataset(data_test_path,
                                  feat_test_path,
                                  lgb_test_path,
                                  sample_ind=test_ind,
                                  transform=subavg_transform, #simple_transform,
                                  mode='eval',
                                  device=device
                                  )
    loader_test = DataLoader(dataset_test, batch_size=batch_size)
    loader_eval = DataLoader(dataset_eval, batch_size=1000, shuffle=False)

    if model_dir is None:
        model_dir = f'outputs/{save_name}/model_state.pt'
    model = torch.load(model_dir, map_location=device)
    challenge_metric = ChallengeMetric()

    # Test
    naive_1 = lambda x: torch.ones(x.shape[:-1]) * 0.06

    item = next(iter(loader_test))
    preds = {
        'naive1': naive_1(item['lc']),
        'normal_1000ppm': torch.normal(item['target'], 1e-3),
        'model': model((item['lc'], item['feat']))
    }

    for name, pred in preds.items():
        print(
            name,
            f"\t{challenge_metric.score(item['target'].cpu(), pred.cpu()).item():.2f}"
        )

    # Evaluation
    preds = []
    print('Evaluate length', len(loader_eval))
    for k, item in tqdm(enumerate(loader_eval)):
        preds += [model((item['lc'], item['feat'])).detach().cpu().numpy()]

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
