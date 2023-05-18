import os
import numpy as np

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from sklearn import metrics
from collections import OrderedDict


def dump_yaml(cfg, path, name=None):
    if name is None:
        name = 'config.yaml'
    else:
        name = name + '.yaml'

    OmegaConf.save(cfg, os.path.join(path, name))
    loaded = OmegaConf.load(os.path.join(path, name))
    assert cfg == loaded


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


def load_pl_state_dict(state_dict, prefix='model'):
    new_state_dict = {k[len(prefix)+1:] :v for k, v in state_dict.items() if k.split('.')[0] == prefix}
    return new_state_dict


def load_state_dict_with_mismatch(model, state_dict):
    model_dict = model.state_dict()

    for name in model_dict.keys():
        if name not in state_dict:
            print(f"Missing keys: {name} while loading checkpoint")

    for name, param in state_dict.items():
        if name not in model_dict:
            print(f"Unexpected keys: {name} while loading checkpoint")
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if model_dict[name].size() != param.size():
            print(f"Ignoring size mismatch for {name} while loading checkpoint")
            continue
        model_dict[name].copy_(param)


def padded_cmap(y_true, y_pred, padding_factor=5):
    num_classes = y_true.shape[1]
    pad_rows = np.ones((padding_factor, num_classes))
    
    y_true = np.concatenate((y_true, pad_rows), axis=0)
    y_pred = np.concatenate((y_pred, pad_rows), axis=0)
    
    score = metrics.average_precision_score(y_true, y_pred, average='macro')

    return score


def mixup(data, targets, alpha=0.4):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


def new_mixup(inputs, targets):
    indices = torch.randperm(inputs.size(0))
    shuffled_data = inputs[indices]
    shuffled_targets = targets[indices]

    lam = np.random.uniform(0.3, 0.7)
    new_data = inputs * lam + shuffled_data * (1 - lam)
    new_targets = targets + shuffled_targets
    new_targets = new_targets.clamp(max=1.0)
    return new_data, new_targets


def mixup_criterion(preds, new_targets, criterion):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


def get_criterion(criterion_type):
    criterions = {
        '2way': BCEFocal2WayLoss(),
        'focal': BCEFocalLoss(),
        'ce': nn.CrossEntropyLoss(label_smoothing=0.01),
        'bce': nn.BCEWithLogitsLoss(),
    }
    return criterions[criterion_type]


def get_activation(criterion_type):
    activations = {
        '2way': nn.Sigmoid(),
        'focal': nn.Sigmoid(),
        'ce': nn.Softmax(dim=-1),
        'bce': nn.Sigmoid(),
    }
    return activations[criterion_type]

# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        if type(preds) == dict:
            preds = preds["logit"]
        
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probs = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1. - probs)**self.gamma * bce_loss
            + (1. - targets) * probs**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()
        self.focal = BCEFocalLoss()
        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target.float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss