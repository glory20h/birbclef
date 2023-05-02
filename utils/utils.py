import os
import torch
import numpy as np

from sklearn import metrics
from collections import OrderedDict


def get_last_checkpoint(path):
    ckpt = None
    contents = os.listdir(path)
    if "last.ckpt" in contents:
        ckpt = os.path.join(path, "last.ckpt")

    return ckpt


def unwrap_checkpoints(exp_path, prefix="model"):
    def unwrap(ckpt, new_name, prefix):
        state = torch.load(ckpt, map_location="cpu")
        state_dict = OrderedDict({k[len(prefix)+1:]: v for k, v in state['state_dict'].items() if k.split('.')[0] == prefix})
        torch.save(state_dict, os.path.join(exp_path, new_name))

    for file in os.listdir(exp_path):
        if ".ckpt" in file and not "unwrapped_" in file:
            unwrap(os.path.join(exp_path, file), "unwrapped_" + file, prefix)


def load_pl_state_dict(ckpt_path, prefix='model'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = {k[len(prefix)+1:] :v for k, v in ckpt['state_dict'].items() if 'pos_embed' not in k}
    return new_state_dict


def padded_cmap(y_true, y_pred, padding_factor=5):
    num_classes = y_true.shape[1]
    pad_rows = np.ones((padding_factor, num_classes))
    
    y_true = np.concatenate((y_true, pad_rows), axis=0)
    y_pred = np.concatenate((y_pred, pad_rows), axis=0)
    
    score = metrics.average_precision_score(y_true, y_pred, average='macro')

    return score