import torch
import numpy as np
from constants import *
from torch.utils.data.dataloader import DataLoader
from utils import ArielMLDataset, ChallengeMetric, simple_transform
import datetime
from tqdm import tqdm
import time
import argparse


def prediction(model_dir='outputs/model_state.pt', save_name='MLP'):
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

    model = torch.load(model_dir)
    challenge_metric = ChallengeMetric()
    print(model.eval())

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
        item['lc'] = item['lc'].to(device)
        preds += [model(item['lc'])]

    eval_pred = torch.cat(preds).detach().cpu().numpy()
    print(eval_pred.shape)

    save_path = f'outputs/{save_name}/evaluation_{datetime.datetime.today().date()}.txt'
    if save_path and (53900, 55) == eval_pred.shape:
        np.savetxt(save_path, eval_pred, fmt='%.10f', delimiter='\t')


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='MLP')
    args = parser.parse_args()
    prediction(save_name=args.save_name)
    print(f'Inference time: {(time.time()-start_time)/60:.3f} mins')
