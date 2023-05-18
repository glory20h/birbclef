import os
import sys
import json
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
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from data import BirbDataset, prepare_data_with_no_labels
from audio_mae import models_mae
from utils import (
    dump_yaml,
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    load_state_dict_with_mismatch,
)


class PreTrainingModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.model = models_mae.MaskedAutoencoderViT(
            patch_size=16, 
            embed_dim=cfg.embed_dim, 
            depth=cfg.depth, 
            num_heads=cfg.num_heads,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            in_chans=1,
            audio_exp=True,
            img_size=(self.cfg.img_len, cfg.n_mels),
            decoder_mode=1,
            decoder_depth=cfg.decoder_depth,
        )
        if cfg.init_ckpt is not None:
            ckpt = torch.load(cfg.init_ckpt, map_location='cpu')
            load_state_dict_with_mismatch(self.model, ckpt['model'])

        self.train_data = BirbDataset(cfg, train_df, use_labels=False)
        self.eval_data = BirbDataset(cfg, val_df, augment=False, use_labels=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.01,
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
        loss, pred, mask, _ = self.model(batch.unsqueeze(1), mask_ratio=self.cfg.mask_ratio)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, mask, _ = self.model(batch.unsqueeze(1), mask_ratio=self.cfg.mask_ratio)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, pred, mask, _ = self.model(batch.unsqueeze(1), mask_ratio=self.cfg.mask_ratio)

        self.log("val_loss", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.cfg.ngpus*8,
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
    parser.add_argument('-c', '-cfg', '--config', default='conf/pt.yaml', help='config path for training')
    parser.add_argument('-t', '--test', action='store_true', help='whether to run the script in testing mode')
    parser.add_argument('--test_ckpt', help='pl ckpt path for testing')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = OmegaConf.load(cfg_path)

    train_df, val_df = prepare_data(cfg, labels=False)

    model = PreTrainingModule(cfg, train_df, val_df)

    os.makedirs(cfg.exp_dir, exist_ok=True)
    dump_yaml(cfg, cfg.exp_dir)

    ckpt_callback = ModelCheckpoint(
        dirpath=cfg.exp_dir,
        filename='checkpoint-{epoch:02d}',
        verbose=True,
        save_last=True,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = Trainer(
        default_root_dir=cfg.exp_dir,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        accelerator="gpu",
        strategy="ddp" if cfg.ngpus > 1 else "auto",
        devices=cfg.ngpus,
        # precision=16 if cfg.use_fp16 else 32,  # 16-mixed?
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