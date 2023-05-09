import os
import math
import random
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from functools import partial
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from audio_mae import models_mae
from models import TimmSED
from data import BirbDataset, prepare_data
from leaf_pytorch.frontend import Leaf
from utils import (
    dump_yaml,
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    load_state_dict_with_mismatch,
    padded_cmap,
    mixup,
    mixup_criterion,
    get_criterion,
    get_activation,
)


class FineTuningModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.model = TimmSED(cfg)
        if cfg.init_ckpt is not None:
            ckpt = torch.load(cfg.init_ckpt, map_location="cpu")
            if 'pytorch-lightning_version' in ckpt:
                ckpt = load_pl_state_dict(ckpt['state_dict'])
                
            load_state_dict_with_mismatch(self.model, ckpt)

        self.criterion = get_criterion(cfg.criterion)
        self.activation = get_activation(cfg.criterion)

        self.train_data = BirbDataset(cfg, train_df)
        self.eval_data = BirbDataset(cfg, val_df, augment=False)

        self.validation_step_logits = []
        self.validation_step_targets = []
        self.test_step_logits = []
        self.test_step_targets = []

        self.leaf = None
        if cfg.feat == "leaf":
            self.leaf = Leaf(
                n_filters=cfg.n_mels,
                sample_rate=cfg.sample_rate,
                window_len=25.0,
                window_stride=10.,
                init_min_freq=cfg.fmin,
                init_max_freq=cfg.fmax,
            )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=1e-4,
        )

    def log(self, name, value, on_step=None, on_epoch=None):
        super().log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        if self.leaf is not None:
            inputs = self.leaf(inputs.unsqueeze(1))

        if random.random() > self.cfg.mixup_prob:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            outputs = self.model(inputs)
            loss = mixup_criterion(outputs, new_targets, self.criterion)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        if self.leaf is not None:
            inputs = self.leaf(inputs.unsqueeze(1))

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.log("val_loss", loss)

        self.validation_step_logits.append(outputs['logit'])
        self.validation_step_targets.append(targets)

    def on_validation_epoch_end(self):
        logits = torch.cat(self.validation_step_logits)
        targets = torch.cat(self.validation_step_targets)

        probs = torch.sigmoid(logits).cpu().detach().numpy()
        targets = targets.cpu().numpy()

        cmap_pad_5 = padded_cmap(targets, probs)
        cmap_pad_1 = padded_cmap(targets, probs, padding_factor=1)
        acc = accuracy_score(np.argmax(targets, axis=-1), np.argmax(probs, axis=-1))
        f1 = f1_score(targets, probs > 0.5, average='micro')

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)
        self.log("acc", acc)
        self.log("f1", f1)

        self.validation_step_logits.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        if self.leaf is not None:
            inputs = self.leaf(inputs.unsqueeze(1))

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        self.log("test_loss", loss)

        self.test_step_logits.append(outputs['logit'])
        self.test_step_targets.append(targets)

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_step_logits)
        targets = torch.cat(self.test_step_targets)

        probs = torch.sigmoid(logits).cpu().detach().numpy()
        targets = targets.cpu().numpy()

        cmap_pad_5 = padded_cmap(targets, probs)
        cmap_pad_1 = padded_cmap(targets, probs, padding_factor=1)

        acc = accuracy_score(np.argmax(targets, axis=-1), np.argmax(probs, axis=-1))
        f1 = f1_score(targets, probs > 0.5, average='micro')

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)
        self.log("acc", acc)
        self.log("f1", f1)

        self.test_step_logits.clear()
        self.test_step_targets.clear()

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.cfg.ngpus*4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data, 
            batch_size=self.batch_size, 
            num_workers=self.cfg.ngpus*4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_data, 
            batch_size=self.batch_size, 
            num_workers=self.cfg.ngpus*4,
        )


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '-cfg', '--config', default='conf/cnn.yaml', help='config path for training')
    parser.add_argument('-t', '--test', action='store_true', help='whether to run the script in testing mode')
    parser.add_argument('--test_ckpt', help='pl ckpt path for testing')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = OmegaConf.load(cfg_path)

    train_df, val_df = prepare_data(cfg)

    model = FineTuningModule(cfg, train_df, val_df)

    os.makedirs(cfg.exp_dir, exist_ok=True)
    dump_yaml(cfg, cfg.exp_dir)

    ckpt_callback = ModelCheckpoint(
        dirpath=cfg.exp_dir,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=2,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=cfg.exp_dir,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        accelerator="gpu",
        # strategy="ddp_find_unused_parameters_true" if cfg.ngpus > 1 else "auto",
        strategy="ddp" if cfg.ngpus > 1 else "auto",
        devices=cfg.ngpus,
        precision=16 if cfg.use_fp16 else 32,
        max_epochs=cfg.epochs,
        callbacks=[ckpt_callback, CustomProgressBar()],
    )

    if args.test:
        trainer.test(model, ckpt_path=args.test_ckpt)
    else:
        trainer.fit(model, ckpt_path=get_last_checkpoint(cfg.exp_dir))

    unwrap_checkpoints(cfg.exp_dir)

if __name__ == '__main__':
    main()