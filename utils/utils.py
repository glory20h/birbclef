import os
import torch

from collections import OrderedDict

def unwrap_checkpoints(exp_path, prefix="model"):
    def unwrap(ckpt, new_name, prefix):
        state = torch.load(ckpt, map_location="cpu")
        state_dict = OrderedDict({k[len(prefix)+1:]: v for k, v in state['state_dict'].items() if k.split('.')[0] == prefix})
        torch.save(state_dict, os.path.join(exp_path, new_name))

    for file in os.listdir(exp_path):
        if ".ckpt" in file and not "unwrapped_" in file:
            unwrap(os.path.join(exp_path, file), "unwrapped_" + file, prefix)

