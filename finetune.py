import os
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

from sklearn.metrics import accuracy_score

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from audio_mae import models_mae
from data import BirbDataset, prepare_data
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
        )
        self.remove_decoder()

        self.clf_head = nn.Sequential(
            nn.Dropout(0.1),
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
        if 'pretrained.pth' in ckpt:
            ckpt = torch.load(ckpt, map_location="cpu")
            load_state_dict_with_mismatch(self.model, ckpt['model'])
            return

        # if stage == 1:
        #     state_dict = load_pl_state_dict(ckpt)
        #     self.model.load_state_dict(state_dict, strict=False)
        # elif stage == 2:
        #     ckpt = torch.load(ckpt, map_location='cpu')
        #     new_state_dict = {k[6:] :v for k, v in ckpt['state_dict'].items() if 'pos_embed' not in k}
        #     del new_state_dict['clf_head.3.weight']
        #     del new_state_dict['clf_head.3.bias']
        #     self.load_state_dict(new_state_dict, strict=False)

        state_dict = torch.load(cfg.init_ckpt, map_location="cpu")
        if 'pytorch-lightning_version' in state_dict:
            state_dict = load_pl_state_dict(state_dict['state_dict'])

        if stage == 1:
            load_state_dict_with_mismatch(self.model, state_dict)
        elif stage == 2:
            load_state_dict_with_mismatch(self, state_dict)
        
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
        self.criterion = get_criterion(cfg.criterion)
        self.activation = get_activation(cfg.criterion)

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
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        if random.random() > self.cfg.mixup_prob:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            outputs = self.model(inputs, self.cfg.mask_t_prob, self.cfg.mask_f_prob)
            loss = mixup_criterion(outputs, new_targets, self.criterion)
        else:
            outputs = self.model(inputs, self.cfg.mask_t_prob, self.cfg.mask_f_prob)
            loss = self.criterion(outputs, targets)

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[0].unsqueeze(1)
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        outputs = self.model(inputs, 0, 0)
        loss = self.criterion(outputs, targets)

        self.log("val_loss", loss)

        self.validation_step_logits.append(outputs)
        self.validation_step_targets.append(targets)

    def on_validation_epoch_end(self):
        logits = torch.cat(self.validation_step_logits)
        targets = torch.cat(self.validation_step_targets)

        probs = self.activation(logits).detach().cpu().numpy()
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
        targets = F.one_hot(batch[1], num_classes=self.cfg.num_classes).float()

        outputs = self.model(inputs, 0, 0)
        loss = self.criterion(outputs, targets)

        self.log("test_loss", loss)

        self.test_step_logits.append(outputs)
        self.test_step_targets.append(targets)

    def on_test_epoch_end(self):
        logits = torch.cat(self.test_step_logits)
        targets = torch.cat(self.test_step_targets)

        probs = self.activation(logits).detach().cpu().numpy()
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