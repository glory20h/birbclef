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
from utils import (
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    padded_cmap,
)


class BirbDataset(Dataset):
    def __init__(self, dataframe, target_len):
        self.df = dataframe
        
        self.melbins = 128
        self.target_len = target_len
        
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
        n_frames = fbank.shape[0]
        p = self.target_len - n_frames
        
        if p > 0:
            # repeat
            # m = torch.nn.ZeroPad2d((0, 0, 0, p))
            # fbank = m(fbank)
            fbank = fbank.repeat(math.ceil(target_len / n_frames), 1)[:target_len]
        elif p < 0:
            # cut
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
        x = self.norm_fbank(fb)
        x = fb.unsqueeze(0)
        
        # target = torch.zeros(NUM_CLASSES)
        # target[row.label + row.label2] = 1
        
        target = row.label
        
        return x, target

    def __len__(self):
        return len(self.df)


class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = models_mae.MaskedAutoencoderViT(
            patch_size=16, 
            embed_dim=768, 
            depth=cfg.depth, 
            num_heads=12,
            mlp_ratio=4, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            in_chans=1,
            audio_exp=True,
            img_size=(cfg.target_len, 128),
            decoder_mode=1,
            decoder_depth=16,
        )
        self.remove_decoder()

        self.clf_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, cfg.num_classes),
        )

        if cfg.init_ckpt is not None:
            self.load_ckpt(cfg.init_ckpt, cfg.ft_stage)

    def remove_decoder(self):
        self.model.mask_token = None
        self.model.decoder_pos_embed = None
        self.model.decoder_embed = None
        self.model.decoder_blocks = None
        self.model.decoder_norm = None
        self.model.decoder_pred = None

    def load_ckpt(self, ckpt, stage):
        if stage == 1:
            state_dict = load_pl_state_dict(ckpt)
            self.model.load_state_dict(state_dict, strict=False)
        elif stage == 2:
            ckpt = torch.load(ckpt, map_location='cpu')
            new_state_dict = {k[6:] :v for k, v in ckpt['state_dict'].items() if 'pos_embed' not in k}
            del new_state_dict['clf_head.3.weight']
            del new_state_dict['clf_head.3.bias']
            self.load_state_dict(new_state_dict, strict=False)
        
    def forward(self, x, mask_t_prob, mask_f_prob):
        x = self.model.enc_forward(x, mask_t_prob, mask_f_prob)
        x = self.clf_head(x[:, 0, :])
        
        return x


class FineTuningModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.model = ModelWrapper(cfg)

        self.train_data = BirbDataset(train_df, cfg.target_len)
        self.eval_data = BirbDataset(val_df, cfg.target_len)

        self.validation_step_logits = []
        self.validation_step_targets = []

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
        x = self.model(batch[0], self.cfg.mask_t_prob, self.cfg.mask_f_prob)
        loss = F.cross_entropy(x, batch[1])
        # loss = F.binary_cross_entropy_with_logits(x, batch[1])

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = self.model(batch[0], 0, 0)
        loss = F.cross_entropy(x, batch[1])
        # loss = F.binary_cross_entropy_with_logits(x, batch[1])

        self.log("val_loss", loss)

        self.validation_step_logits.append(x)
        self.validation_step_targets.append(batch[1])

    def on_validation_epoch_end(self):
        pred = torch.cat(self.validation_step_logits)
        target = torch.cat(self.validation_step_targets)

        pred = F.softmax(pred, dim=-1).cpu().detach().numpy()
        target = F.one_hot(target, num_classes=self.cfg.num_classes).cpu().numpy()
        # target = target.cpu().numpy()

        cmap_pad_5 = padded_cmap(target, pred, self.cfg.num_classes)
        cmap_pad_1 = padded_cmap(target, pred, self.cfg.num_classes, padding_factor=1)

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)

        self.validation_step_logits.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        x = self.model(batch[0], 0, 0)
        loss = F.cross_entropy(x, batch[1])
        # loss = F.binary_cross_entropy_with_logits(x, batch[1])

        self.log("val_loss", loss)

        self.test_step_logits.append(x)
        self.tset_step_targets.append(batch[1])

    def on_test_epoch_end(self):
        pred = torch.cat(self.test_step_logits)
        target = torch.cat(self.test_step_targets)

        pred = F.softmax(pred, dim=-1).cpu().detach().numpy()
        target = F.one_hot(target, num_classes=self.cfg.num_classes).cpu().numpy()
        # target = target.cpu().numpy()

        cmap_pad_5 = padded_cmap(target, pred, self.cfg.num_classes)
        cmap_pad_1 = padded_cmap(target, pred, self.cfg.num_classes, padding_factor=1)

        self.log("cmAP5", cmap_pad_5)
        self.log("cmAP1", cmap_pad_1)

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
    parser.add_argument('-c', '-cfg', '--config', help='config path for training')
    parser.add_argument('-t', '--test', action='store_true', help='whether to run the script in testing mode')
    parser.add_argument('--test_ckpt', help='pl ckpt path for testing')
    args = parser.parse_args()

    cfg_path = args.config
    cfg = OmegaConf.load(cfg_path)

    # Data Preparation
    # class_names = sorted(os.listdir(os.path.join(BIRB_23_PATH, 'train_audio')))
    # class_labels = list(range(len(class_names)))

    # n2l = dict(zip(class_names, class_labels))

    # df_23 = pd.read_csv(f'{BIRB_23_PATH}/train_metadata.csv')
    # df_23['filepath'] = BIRB_23_PATH + '/train_audio/' + df_23.filename
    # df_23['label'] = df_23.primary_label.map(lambda x: [n2l[x]])
    # df_23['label2'] = df_23.secondary_labels.map(lambda x: [n2l[n] for n in eval(x)])
    # df_23 = df_23[['filepath', 'label', 'label2']]

    # # skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=2023)
    # # df_23["fold"] = -1
    # # for fold, (train_idx, val_idx) in enumerate(skf.split(df_23, df_23['label'])):
    # #     df_23.loc[val_idx, 'fold'] = fold

    # # train_df = df_23.query("fold!=0").reset_index(drop=True)
    # # valid_df = df_23.query("fold==0").reset_index(drop=True)

    # train_df, val_df = train_test_split(df_23, test_size=0.05, random_state=2023)

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

    if cfg.ft_stage == 1:
        df_all = pd.concat([df_20, df_21, df_22, df_xc], axis=0, ignore_index=True)
        nodup_idx = df_all[['xc_id']].drop_duplicates().index
        df_all = df_all.loc[nodup_idx].reset_index(drop=True)
        corrupt_files = json.load(open('corrupt_files.json', 'r'))
        df_all = df_all[~df_all.filename.isin(corrupt_files)]
        corrupt_files = json.load(open('corrupt_files2.json', 'r'))
        df_all = df_all[~df_all.filename.isin(corrupt_files)]
        df_all = df_all.reset_index(drop=True)

        class_names = sorted(list(df_all.primary_label.unique()))
        class_labels = list(range(len(class_names)))
        n2l = dict(zip(class_names, class_labels))
        df_all['label'] = df_all.primary_label.map(lambda x: n2l[x])

    elif cfg.ft_stage == 2:
        class_names = sorted(os.listdir(os.path.join(cfg.birb_23_path, 'train_audio')))
        class_labels = list(range(len(class_names)))
        n2l = dict(zip(class_names, class_labels))

        df_all = pd.concat([df_20, df_21, df_22, df_xc, df_23], axis=0, ignore_index=True)
        nodup_idx = df_all[['xc_id']].drop_duplicates().index
        df_all = df_all.loc[nodup_idx].reset_index(drop=True)
        corrupt_files = json.load(open('corrupt_files.json', 'r'))
        df_all = df_all[~df_all.filename.isin(corrupt_files)]
        corrupt_files = json.load(open('corrupt_files2.json', 'r'))
        df_all = df_all[~df_all.filename.isin(corrupt_files)]
        df_all = df_all[df_all.primary_label.isin(class_names)]
        df_all = df_all.reset_index(drop=True)

        df_all['label'] = df_all.primary_label.map(lambda x: n2l[x])

    skf = StratifiedKFold(n_splits=20, shuffle=True, random_state=2023)
    df_all["fold"] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, df_all['label'])):
        df_all.loc[val_idx, 'fold'] = fold

    train_df = df_all.query("fold!=0").reset_index(drop=True)
    val_df = df_all.query("fold==0").reset_index(drop=True)

    model = FineTuningModule(cfg, train_df, val_df)

    os.makedirs(cfg.exp_dir, exist_ok=True)

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
        strategy="ddp" if cfg.ngpus > 1 else "auto",
        devices=cfg.ngpus,
        precision=16 if cfg.use_fp16 else 32,
        max_epochs=cfg.epochs,
        callbacks=[ckpt_callback, CustomProgressBar()],
    )

    if args.test:
        trainer.test(model, ckpt_path=args.ckpt_path)
    else:
        trainer.fit(model, ckpt_path=get_last_checkpoint(cfg.exp_dir))

    unwrap_checkpoints(cfg.exp_dir)

if __name__ == '__main__':
    main()