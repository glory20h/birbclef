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
from utils import (
    dump_yaml,
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    load_state_dict_with_mismatch,
)


class BirbDataset(Dataset):
    def __init__(self, cfg, dataframe, augment=True):
        self.df = dataframe
        self.duration = cfg.duration
        self.melbins = cfg.melbins
        self.augment = augment
        
    def wav2fbank(self, audio, sr):
        fbank = kaldi.fbank(
            audio.unsqueeze(0), 
            htk_compat=True, 
            sample_frequency=sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.melbins, 
            dither=0.0, 
            frame_shift=10
        )
        return fbank
    
    def load_audio(self, filename, target_sr=32000):
        audio, sr = torchaudio.load(filename)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
        return audio[0], sr
    
    def cut_or_repeat(self, audio, sr):
        target_len = int(sr * self.duration)
        p = target_len - len(audio)

        if p > 0:
            audio = audio.repeat(math.ceil(target_len / len(audio)))[:target_len]
        elif p < 0:
            i = random.randint(0, -p)
            audio = audio[i:i+target_len]

        return audio
    
    def normalize(self, x):
        x = x - x.mean()
        return x / x.abs().max()
    
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
        
        audio, sr = self.load_audio(filename)
        audio = self.cut_or_repeat(audio, sr)
        audio = self.normalize(audio)
            
        fb = self.wav2fbank(audio, sr)
        
        return fb

    def __len__(self):
        return len(self.df)


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
            img_size=(1024, 128),
            decoder_mode=1,
            decoder_depth=16,
        )
        if cfg.ckpt is not None:
            ckpt = torch.load(cfg.ckpt, map_location='cpu')
            try:
                self.model.load_state_dict(ckpt['model'], strict=False)
            except:
                load_state_dict_with_mismatch(self.model, ckpt['model'])

        self.train_data = BirbDataset(cfg, train_df)
        self.eval_data = BirbDataset(cfg, val_df, augment=False)

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

        if torch.isnan(loss):
            torch.save(batch, 'prob_batch.pt')
            torch.save(self.model.state_dict(), 'prob_ckpt.pt')

            raise Exception("Nan loss?")

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


def prepare_data(cfg):
    df_20 = pd.read_csv(os.path.join(cfg.birb_20_path, 'train.csv'))
    df_20['filepath'] = cfg.birb_20_path + '/train_audio/' + df_20.ebird_code + '/' + df_20.filename
    df_20['xc_id'] = df_20.filename.map(lambda x: x.split('.')[0])
    df_20 = df_20[['filepath', 'filename', 'ebird_code', 'xc_id']]
    df_20.rename(columns={"ebird_code": "primary_label"}, inplace=True)
    df_20['birdclef'] = '20'

    df_21 = pd.read_csv(os.path.join(cfg.birb_21_path, 'train_metadata.csv'))
    df_21['filepath'] = cfg.birb_21_path + '/train_short_audio/' + df_21.primary_label + '/' + df_21.filename

    corrupt_paths = [cfg.birb_21_path + '/train_short_audio/houwre/XC590621.ogg',
                    cfg.birb_21_path + '/train_short_audio/cogdov/XC579430.ogg']
    df_21 = df_21[~df_21.filepath.isin(corrupt_paths)]

    df_21['xc_id'] = df_21.filename.map(lambda x: x.split('.')[0])
    df_21 = df_21[['filepath', 'filename', 'primary_label', 'xc_id']]
    df_21['birdclef'] = '21'

    df_22 = pd.read_csv(os.path.join(cfg.birb_22_path, 'train_metadata.csv'))
    df_22['filepath'] = cfg.birb_22_path + '/train_audio/' + df_22.filename
    df_22['filename'] = df_22.filename.map(lambda x: x.split('/')[-1])
    df_22['xc_id'] = df_22.filename.map(lambda x: x.split('.')[0])
    df_22 = df_22[['filepath', 'filename', 'primary_label', 'xc_id']]
    df_22['birdclef'] = '22'

    df_xam = pd.read_csv(os.path.join(cfg.xc_am_path, 'train_extended.csv'))
    df_xam = df_xam[:14685]
    df_xam['filepath'] = cfg.xc_am_path + '/A-M/' + df_xam.ebird_code + '/' + df_xam.filename

    df_xnz = pd.read_csv(os.path.join(cfg.xc_nz_path, 'train_extended.csv'))
    df_xnz = df_xnz[14685:]
    df_xnz['filepath'] = cfg.xc_nz_path + '/N-Z/' + df_xnz.ebird_code + '/' + df_xnz.filename

    df_xc = pd.concat([df_xam, df_xnz], axis=0, ignore_index=True)
    df_xc['xc_id'] = df_xc.filename.map(lambda x: x.split('.')[0])
    df_xc = df_xc[['filepath', 'filename', 'ebird_code', 'xc_id']]
    df_xc.rename(columns={"ebird_code": "primary_label"}, inplace=True)
    df_xc['birdclef'] = 'xc'

    df_23 = pd.read_csv(f'{cfg.birb_23_path}/train_metadata.csv')
    df_23['filepath'] = cfg.birb_23_path + '/train_audio/' + df_23.filename
    df_23['filename'] = df_23.filename.map(lambda x: x.split('/')[-1])
    df_23['xc_id'] = df_23.filename.map(lambda x: x.split('.')[0])
    df_23 = df_23[['filepath', 'filename', 'primary_label', 'xc_id']]
    df_23['birdclef'] = '23'

    df_all = pd.concat([df_20, df_21, df_22, df_23, df_xc], axis=0, ignore_index=True)
    nodup_idx = df_all[['xc_id']].drop_duplicates().index
    df_all = df_all.loc[nodup_idx].reset_index(drop=True)
    corrupt_files = json.load(open('corrupt_files.json', 'r'))
    df_all = df_all[~df_all.filename.isin(corrupt_files)]
    corrupt_files = json.load(open('corrupt_files2.json', 'r'))
    df_all = df_all[~df_all.filename.isin(corrupt_files)]
    df_all = df_all.reset_index(drop=True)

    # split the data into train and validation sets
    return train_test_split(df_all, test_size=0.05, random_state=2023)

def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '-cfg', '--config', default='conf/pt.yaml', help='config path for training')
    parser.add_argument('-t', '--test', action='store_true', help='whether to run the script in testing mode')
    parser.add_argument('--test_ckpt', help='pl ckpt path for testing')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = OmegaConf.load(cfg_path)

    train_df, val_df = prepare_data(cfg)

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