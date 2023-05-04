import os
import numpy as np
import pandas as pd

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


def load_pl_state_dict(ckpt_path, prefix='model'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = {k[len(prefix)+1:] :v for k, v in ckpt['state_dict'].items() if 'pos_embed' not in k}
    return new_state_dict


def load_state_dict_with_mismatch(model, state_dict):
    model_dict = model.state_dict()

    for name, param in state_dict.items():
        if name not in model_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        if model_dict[name].size() != param.size():
            print(f"Ignoring size mismatch for {name} while loading checkpoint")
            continue
        model_dict[name].copy_(param)

    return model


def padded_cmap(y_true, y_pred, padding_factor=5):
    num_classes = y_true.shape[1]
    pad_rows = np.ones((padding_factor, num_classes))
    
    y_true = np.concatenate((y_true, pad_rows), axis=0)
    y_pred = np.concatenate((y_pred, pad_rows), axis=0)
    
    score = metrics.average_precision_score(y_true, y_pred, average='macro')

    return score


def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    

def upsample_data(df, thr=20, seed=2023):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df


def downsample_data(df, thr=500, seed=2023):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


# https://www.kaggle.com/c/rfcx-species-audio-detection/discussion/213075
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1. - probas)**self.gamma * bce_loss
            + (1. - targets) * probas**self.gamma * bce_loss
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