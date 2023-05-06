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

import torchaudio

from sklearn.metrics import accuracy_score

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from audio_mae import models_mae
from models import TimmSED
from data import prepare_data
from utils import (
    dump_yaml,
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    load_state_dict_with_mismatch,
    padded_cmap,
    mixup,
    BCEFocal2WayLoss,
)

class BirbDataset2(Dataset):
    def __init__(self, cfg, dataframe, augment=True):
        self.df = dataframe
        self.duration = cfg.duration
        self.augment = augment
    
    def load_audio(self, filename, target_sr=32000):
        audio, sr = torchaudio.load(filename)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
        return audio[0], sr
    
    def cut_or_repeat(self, audio, sr):
        target_len = sr * self.duration
        p = target_len - len(audio)

        if p > 0:
            audio = audio.repeat(math.ceil(target_len / len(audio)))[:target_len]
        elif p < 0:
            i = random.randint(0, -p)
            audio = audio[i:i+target_len]

        return audio
    
    def add_gaussian_noise(self, audio, min_snr=5, max_snr=20):
        snr = np.random.uniform(min_snr, max_snr)

        a_signal = torch.sqrt(audio ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = torch.randn(len(audio))
        a_white = torch.sqrt(white_noise ** 2).max()
        audio = audio + white_noise * 1 / a_white * a_noise

        return audio

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row.filepath
        label = row.label
        
        audio, sr = self.load_audio(filename)
        audio = self.cut_or_repeat(audio, sr)
        if self.augment and random.random() > 0.5:
            audio = self.add_gaussian_noise(audio)
        
        return audio, label

    def __len__(self):
        return len(self.df)


criterion = BCEFocal2WayLoss()
def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)


class FineTuningModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.model = TimmSED(cfg)

        self.train_data = BirbDataset2(cfg, train_df)
        self.eval_data = BirbDataset2(cfg, val_df, augment=False)

        self.validation_step_logits = []
        self.validation_step_targets = []
        self.test_step_logits = []
        self.test_step_targets = []

        if cfg.init_ckpt is not None:
            ckpt = torch.load(cfg.init_ckpt, map_location="cpu")
            if 'pytorch-lightning_version' in ckpt:
                ckpt = load_pl_state_dict(ckpt['state_dict'])
                
            load_state_dict_with_mismatch(self.model, ckpt)

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
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        if random.random() > self.cfg.mixup_prob:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            outputs = self.model(inputs)
            loss = mixup_criterion(outputs, new_targets)
        else:
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        self.log("val_loss", loss)

        self.validation_step_logits.append(outputs['logit'])
        self.validation_step_targets.append(targets)

    def on_validation_epoch_end(self):
        logits = torch.cat(self.validation_step_logits)
        targets = torch.cat(self.validation_step_targets)

        probs = torch.sigmoid(logits).cpu().detach().numpy()
        target = targets.cpu().numpy()

        cmap_pad_5 = padded_cmap(target, probs)
        cmap_pad_1 = padded_cmap(target, probs, padding_factor=1)

        acc = accuracy_score(np.argmax(target, axis=-1), np.argmax(probs, axis=-1))

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)
        self.log("acc", acc)

        self.validation_step_logits.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        inputs = batch[0]
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        self.log("test_loss", loss)

        self.test_step_logits.append(outputs['logit'])
        self.test_step_targets.append(targets)

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_step_logits)
        targets = torch.cat(self.test_step_targets)

        probs = torch.sigmoid(logits).cpu().detach().numpy()
        target = targets.cpu().numpy()

        cmap_pad_5 = padded_cmap(target, probs)
        cmap_pad_1 = padded_cmap(target, probs, padding_factor=1)

        acc = accuracy_score(np.argmax(target, axis=-1), np.argmax(probs, axis=-1))

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)
        self.log("acc", acc)

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