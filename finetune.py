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
from torchaudio.compliance import kaldi

from sklearn.metrics import accuracy_score

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from audio_mae import models_mae
from models import AttBlockV2
from data import prepare_data
from utils import (
    dump_yaml,
    get_last_checkpoint,
    unwrap_checkpoints,
    load_pl_state_dict,
    load_state_dict_with_mismatch,
    padded_cmap,
    mixup,
    BCEFocalLoss,
)


class BirbDataset(Dataset):
    def __init__(self, cfg, dataframe, augment=True):
        self.df = dataframe
        self.duration = cfg.duration
        self.melbins = cfg.melbins
        self.img_len = cfg.img_len
        self.augment = augment
        
        self.spec_transform = None
        if cfg.use_spec:
            self.spec_transform = torchaudio.transforms.Spectrogram(
                n_fft=254,
                hop_length=312,
                normalized=True,
            )

    def wav2fbank(self, audio, sr):
        fbank = kaldi.fbank(
            audio.unsqueeze(0), 
            htk_compat=True, 
            sample_frequency=sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.melbins, 
            dither=0.0, 
            frame_shift=10,
        )
        return fbank
    
    def wav2spec(self, audio):
        spec = self.spec_transform(audio)
        spec = spec.T[:self.img_len]
        return spec
    
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
        
        if self.spec_transform is not None:
            feat = self.wav2spec(audio)
        else:
            feat = self.wav2fbank(audio, sr)
        
        return feat, label

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
            img_size=(cfg.img_len, 128),
            decoder_mode=1,
            decoder_depth=16,
        )
        self.remove_decoder()

        self.clf_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
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
        if 'pretrained.pth' in ckpt:
            ckpt = torch.load(ckpt, map_location="cpu")
            load_state_dict_with_mismatch(self.model, ckpt['model'])
            return

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


criterion = BCEFocalLoss()
def mixup_criterion(preds, new_targets):
    targets1, targets2, lam = new_targets[0], new_targets[1], new_targets[2]
    return lam * criterion(preds, targets1.float()) + (1 - lam) * criterion(preds, targets2.float())


class FineTuningModule(LightningModule):
    def __init__(self, cfg, train_df, val_df):
        super().__init__()
        self.cfg = cfg

        self.batch_size = cfg.batch_size
        self.model = ModelWrapper(cfg)

        self.train_data = BirbDataset(cfg, train_df)
        self.eval_data = BirbDataset(cfg, val_df, augment=False)

        self.validation_step_logits = []
        self.validation_step_targets = []
        self.test_step_logits = []
        self.test_step_targets = []

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
        inputs = batch[0].unsqueeze(1)
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        if random.random() > self.cfg.mixup_prob:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            outputs = self.model(inputs, self.cfg.mask_t_prob, self.cfg.mask_f_prob)
            loss = mixup_criterion(outputs, new_targets)
        else:
            outputs = self.model(inputs, self.cfg.mask_t_prob, self.cfg.mask_f_prob)
            loss = criterion(outputs, targets)

        self.log("loss", loss)

        return loss

        # x = self.model(batch[0].unsqueeze(1), self.cfg.mask_t_prob, self.cfg.mask_f_prob)
        # loss = F.cross_entropy(x, batch[1])

        # self.log("loss", loss)

        # return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0].unsqueeze(1)
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        outputs = self.model(inputs, 0, 0)
        loss = criterion(outputs, targets)

        self.log("val_loss", loss)

        self.validation_step_logits.append(outputs)
        self.validation_step_targets.append(targets)

        # x = self.model(batch[0].unsqueeze(1), 0, 0)
        # loss = F.cross_entropy(x, batch[1])
        # # loss = F.binary_cross_entropy_with_logits(x, batch[1])

        # self.log("val_loss", loss)

        # self.validation_step_logits.append(x)
        # self.validation_step_targets.append(batch[1])

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

        # pred = torch.cat(self.validation_step_logits)
        # target = torch.cat(self.validation_step_targets)

        # probs = F.softmax(pred, dim=-1).cpu().detach().numpy()
        # target = F.one_hot(target, num_classes=self.cfg.num_classes).cpu().numpy()

        # cmap_pad_5 = padded_cmap(target, probs)
        # cmap_pad_1 = padded_cmap(target, probs, padding_factor=1)
        # acc = accuracy_score(np.argmax(target, axis=-1), np.argmax(probs, axis=-1))

        # self.log("cmAP5", cmap_pad_5)
        # self.log("cmAP1", cmap_pad_1)
        # self.log("acc", acc)

        # self.validation_step_logits.clear()
        # self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        inputs = batch[0].unsqueeze(1)
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes)

        outputs = self.model(inputs, 0, 0)
        loss = criterion(outputs, targets)

        self.log("test_loss", loss)

        self.test_step_logits.append(outputs)
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
    parser.add_argument('-c', '-cfg', '--config', help='config path for training')
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