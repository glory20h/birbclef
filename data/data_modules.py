import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance import kaldi

from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation

from data import prepare_noise_data


class TorchLibrosaSpectrogram:
    def __init__(self, cfg):
        self.spectrogram_extractor = Spectrogram(
            n_fft=2048, 
            hop_length=512,
            win_length=2048, 
            window="hann", 
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=cfg.sample_rate, 
            n_fft=2048,
            n_mels=cfg.n_mels, 
            fmin=cfg.fmin, 
            fmax=cfg.fmax, 
            ref=1.0, 
            amin=1e-10, 
            top_db=None,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(cfg.n_mels)

    def __call__(self, audio):
        feat = self.spectrogram_extractor(audio.unsqueeze(0))
        feat = self.logmel_extractor(feat)
        feat = feat.squeeze().T
    
        return feat


class BirbDataset(Dataset):
    def __init__(self, cfg, dataframe, augment=True, use_labels=True):
        self.df = dataframe
        self.duration = cfg.duration
        self.n_mels = cfg.n_mels
        self.img_len = cfg.img_len
        self.target_sr = cfg.sample_rate
        self.num_classes = cfg.num_classes
        self.temporal_label = cfg.temporal_label is not None
        self.augment = augment
        self.use_labels = use_labels
        
        self.spec_transform = None
        if cfg.feat == 'spec':
            self.spec_transform = nn.Sequential(
                torchaudio.transforms.Spectrogram(
                    n_fft=254,
                    hop_length=312,
                    normalized=True,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        elif cfg.feat == 'mel_spec':
            self.spec_transform = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=cfg.sample_rate,
                    n_fft=625,
                    f_min=cfg.fmin,
                    f_max=cfg.fmax,
                    normalized=True,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        elif cfg.feat == 'tl_spec':
            self.spec_transform = TorchLibrosaSpectrogram(cfg)
        elif cfg.feat in ['kaldi_fb', 'leaf', 'audio']:
            self.spec_transform = cfg.feat
        else:
            raise NotImplementedError(
                "'feat' must be one of 'spec', 'mel_spec',"
                " 'tl_spec', 'kaldi_fb', 'leaf, 'audio'."
            )
        
        self.spec_augmenter = None
        if augment:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=64, 
                time_stripes_num=2,
                freq_drop_width=8, 
                freq_stripes_num=2,
            )
            
        self.noise_df = prepare_noise_data(cfg)

    def wav2fbank(self, audio, sr):
        fbank = kaldi.fbank(
            audio.unsqueeze(0), 
            htk_compat=True, 
            sample_frequency=sr, 
            use_energy=False,
            window_type='hanning', 
            num_mel_bins=self.n_mels, 
            dither=0.0, 
            frame_shift=10,
            # low_freq=80, # default 20.0
            # high_freq=16000, # default 0.0
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
            i = 0
        elif p < 0:
            i = random.randint(0, -p)
            audio = audio[i:i+target_len]
            
        audio = audio - audio.mean()
        return audio, i
    
    def add_gaussian_noise(self, audio, min_snr=10, max_snr=20):
        snr = np.random.uniform(min_snr, max_snr)

        a_signal = torch.sqrt(audio ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = torch.randn(len(audio))
        a_white = torch.sqrt(white_noise ** 2).max()
        audio = audio + white_noise * 1 / a_white * a_noise

        return audio
    
    def add_noise(self, audio, min_snr=6, max_snr=20):
        snr = np.random.uniform(min_snr, max_snr)

        a_signal = torch.sqrt(audio ** 2).max()
        a_signal = a_signal / (10 ** (snr / 20))

        noise, sr = self.load_audio(
            self.noise_df.filepath.iloc[random.randint(0, len(self.noise_df)-1)],
            self.target_sr,
        )
        noise, _ = self.cut_or_repeat(noise, sr)
        a_noise = torch.sqrt(noise ** 2).max()
        audio = audio + noise * 1 / a_noise * a_signal

        return audio
    
    def denoise(self, audio, max_thr=0.2, max_scale=0.5):
        thr = random.uniform(0.05, max_thr)
        scale = random.uniform(max_scale, 1)
        thr = ((audio.max() - audio.min()) / 2) * thr
        audio[audio.abs() < thr] *= scale
        return audio
    
    def high_pass(self, audio, sr=32000):
        cutoff_freq = random.uniform(200, 2500)
        return torchaudio.functional.highpass_biquad(
            audio,
            sr,
            cutoff_freq=cutoff_freq,
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]
        filename = row.filepath
        
        audio, sr = self.load_audio(filename, self.target_sr)
        audio, i = self.cut_or_repeat(audio, sr)
        if self.augment:
            if random.random() > 0.5:
                audio = self.denoise(audio)
            if random.random() > 0.8:
                audio = self.high_pass(audio, sr)
            if random.random() > 0.5:
                audio = self.add_gaussian_noise(audio)
            if random.random() > 0.5:
                audio = self.add_noise(audio)

        if self.spec_transform in ["audio", "leaf"]:
            feat = audio
        else:
            if self.spec_transform == "kaldi_fb":
                feat = self.wav2fbank(audio, sr)
            else:
                feat = self.wav2spec(audio)
                
            if self.augment:
                feat = feat.unsqueeze(0).unsqueeze(0)
                feat = self.spec_augmenter(feat).squeeze()

        if self.use_labels:
            p_l = row.label
            if self.temporal_label:
                s_l = row.secondary_labels
                start = i / self.target_sr
                end = start + self.duration
                
                target = np.zeros(self.num_classes)
                for tl in eval(row.temporal_label):
                    l = tl[0]
                    for seg in tl[1:]:
                        if start <= seg[1] and end >= seg[0]:
                            if l in [p_l, *s_l]:
                                target[l] = 1.0
                            else:
                                target[l] = 0.5
            else:
                target = np.zeros(self.num_classes)
                target[p_l] = 1.0
                for l in row.secondary_labels:
                    target[l] = 1.0
            return feat, target

        return feat

    def __len__(self):
        return len(self.df)