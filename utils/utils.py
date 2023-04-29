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


def padded_cmap(solution, submission, num_classes, padding_factor=5):
    pad = np.ones((padding_factor, num_classes))
    
    padded_solution = np.concatenate((solution, pad), axis=0)
    padded_submission = np.concatenate((submission, pad), axis=0)
    
    score = metrics.average_precision_score(
        padded_solution,
        padded_submission,
        average='macro',
    )
    return score