import os
import sys
import json
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
from torchaudio.compliance import kaldi

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from audio_mae import models_mae
from utils import unwrap_checkpoints


class BirbDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        
        self.melbins = 128
        self.target_len = 1024
        
    def wav2fbank(self, filename):
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        # 498 128
        fbank = kaldi.fbank(
            waveform, 
            htk_compat=True, 
            sample_frequency=sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.melbins, 
            dither=0.0, 
            frame_shift=10
        )
        # AudioSet: 1024 (16K sr)
        # ESC: 512 (8K sr)
        
        # cut and pad
        n_frames = fbank.shape[0]
        p = self.target_len - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            i = random.randint(0, -p)
            fbank = fbank[i:i+self.target_len, :]
        
        return fbank

    def norm_fbank(self, fbank):
        norm_mean = -4.2677393
        norm_std = 4.5689974
        fbank = (fbank - norm_mean) / (norm_std * 2)
        return fbank

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fb = self.wav2fbank(row.filepath)
        # What's the effect of "fixed normalization"? Is it needed?
        x = self.norm_fbank(fb)

        x = x.unsqueeze(0)
        
        return x

    def __len__(self):
        return len(self.df)


class PreTrainingModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.model = models_mae.MaskedAutoencoderViT(
            patch_size=16, 
            embed_dim=768, 
            depth=cfg.depth, 
            num_heads=12,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            in_chans=1,
            audio_exp=True,
            img_size=(1024, 128),
            decoder_mode=1,
            decoder_depth=16,
        )
        ckpt = torch.load(cfg.ckpt, map_location='cpu')
        self.model.load_state_dict(ckpt['model'], strict=False)

        self.train_data = BirbDataset(train_df)
        self.eval_data = BirbDataset(val_df)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=LR,
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
        loss, pred, mask, _ = self.model(batch, mask_ratio=cfg.mask_ratio)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred, mask, _ = self.model(batch, mask_ratio=cfg.mask_ratio)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss, pred, mask, _ = self.model(batch, mask_ratio=cfg.mask_ratio)

        self.log("val_loss", loss)

    def train_dataloader(self):
        return DataLoader(
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=cfg.ngpus*8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_data, 
            batch_size=self.batch_size, 
            num_workers=cfg.ngpus*4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.eval_data, 
            batch_size=self.batch_size, 
            num_workers=cfg.ngpus*4,
        )


class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '-cfg', '--config', default='conf/ft.yaml', help='config path for training')
    parser.add_argument('-t', '--test', action='store_true', help='whether to run the script in testing mode')
    parser.add_argument('--test_ckpt', help='pl ckpt path for testing')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = OmegaConf.load(cfg_path)

    ################################# PREPARE DATA #################################
    df_20 = pd.read_csv(os.path.join(cfg.birb_20_path, 'train.csv'))
    df_20['filepath'] = cfg.birb_20_path + '/train_audio/' + df_20.ebird_code + '/' + df_20.filename
    df_20['xc_id'] = df_20.filename.map(lambda x: x.split('.')[0])
    df_20 = df_20[['filepath', 'filename', 'xc_id']]
    df_20['birdclef'] = '20'

    df_21 = pd.read_csv(os.path.join(cfg.birb_21_path, 'train_metadata.csv'))
    df_21['filepath'] = cfg.birb_21_path + '/train_short_audio/' + df_21.primary_label + '/' + df_21.filename

    corrupt_paths = [cfg.birb_21_path + '/train_short_audio/houwre/XC590621.ogg',
                    cfg.birb_21_path + '/train_short_audio/cogdov/XC579430.ogg']
    df_21 = df_21[~df_21.filepath.isin(corrupt_paths)]

    df_21['xc_id'] = df_21.filename.map(lambda x: x.split('.')[0])
    df_21 = df_21[['filepath', 'filename', 'xc_id']]
    df_21['birdclef'] = '21'

    df_22 = pd.read_csv(os.path.join(cfg.birb_22_path, 'train_metadata.csv'))
    df_22['filepath'] = cfg.birb_22_path + '/train_audio/' + df_22.filename
    df_22['filename'] = df_22.filename.map(lambda x: x.split('/')[-1])
    df_22['xc_id'] = df_22.filename.map(lambda x: x.split('.')[0])
    df_22 = df_22[['filepath', 'filename', 'xc_id']]
    df_22['birdclef'] = '22'

    df_23 = pd.read_csv(os.path.join(cfg.birb_23_path, 'train_metadata.csv'))
    df_23['filepath'] = cfg.birb_23_path + '/train_audio/' + df_23.filename
    df_23['filename'] = df_23.filename.map(lambda x: x.split('/')[-1])
    df_23['xc_id'] = df_23.filename.map(lambda x: x.split('.')[0])
    df_23 = df_23[['filepath', 'filename', 'xc_id']]
    df_23['birdclef'] = '23'

    df_xam = pd.read_csv(os.path.join(cfg.xc_am_path, 'train_extended.csv'))
    df_xam = df_xam[:14685]
    df_xam['filepath'] = cfg.xc_am_path + '/A-M/' + df_xam.ebird_code + '/' + df_xam.filename

    df_xnz = pd.read_csv(os.path.join(cfg.xc_nz_path, 'train_extended.csv'))
    df_xnz = df_xnz[14685:]
    df_xnz['filepath'] = cfg.xc_nz_path + '/N-Z/' + df_xnz.ebird_code + '/' + df_xnz.filename

    df_xc = pd.concat([df_xam, df_xnz], axis=0, ignore_index=True)
    df_xc['xc_id'] = df_xc.filename.map(lambda x: x.split('.')[0])
    df_xc = df_xc[['filepath', 'filename', 'xc_id']]
    df_xc['birdclef'] = 'xc'

    df_all = pd.concat([df_20, df_21, df_22, df_23, df_xc], axis=0, ignore_index=True)

    nodup_idx = df_all[['xc_id']].drop_duplicates().index
    df_all = df_all.loc[nodup_idx].reset_index(drop=True)

    corrupt_files = json.load(open('corrupt_files.json', 'r'))
    df_all = df_all[~df_all.filename.isin(corrupt_files)]
    corrupt_files = json.load(open('corrupt_files2.json', 'r'))
    df_all = df_all[~df_all.filename.isin(corrupt_files)]

    ################################# PREPARE DATA #################################

    # split the data into train and validation sets
    train_df, val_df = train_test_split(df_all, test_size=0.05, random_state=2023)

    model = PreTrainingModule(cfg, train_data, eval_data)

    os.makedirs(cfg.exp_dir, exist_ok=True)

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
        strategy="ddp" if cfg.ngpus > 1 else None,
        devices=cfg.ngpus,
        precision=16 if cfg.use_fp16 else 32,  # 16-mixed?
        max_epochs=cfg.epochs,
        callbacks=[ckpt_callback, CustomProgressBar()],
    )

    if args.test:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, ckpt_path=None)


if __name__ == '__main__':
    main()