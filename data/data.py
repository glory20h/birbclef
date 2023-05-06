import os
import json
import math
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.compliance import kaldi

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation


def filter_data(df, thr=5):
    # Count the number of samples for each class
    counts = df.primary_label.value_counts()

    # Condition that selects classes with less than `thr` samples
    cond = df.primary_label.isin(counts[counts<thr].index.tolist())

    # Add a new column to select samples for cross validation
    df['cv'] = True

    # Set cv = False for those class where there is samples less than thr
    df.loc[cond, 'cv'] = False

    # Return the filtered dataframe
    return df
    

# def upsample_data(df, thr=20, seed=2023):
#     # get the class distribution
#     class_dist = df['primary_label'].value_counts()

#     # identify the classes that have less than the threshold number of samples
#     down_classes = class_dist[class_dist < thr].index.tolist()

#     # create an empty list to store the upsampled dataframes
#     up_dfs = []

#     # loop through the undersampled classes and upsample them
#     for c in down_classes:
#         # get the dataframe for the current class
#         class_df = df.query("primary_label==@c")
#         # find number of samples to add
#         num_up = thr - class_df.shape[0]
#         # upsample the dataframe
#         class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
#         # append the upsampled dataframe to the list
#         up_dfs.append(class_df)

#     # concatenate the upsampled dataframes and the original dataframe
#     up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
#     return up_df


def upsample_data(train_df, seed):
    X = train_df.drop(columns=['label'])
    y = train_df['label']

    ros = RandomOverSampler(random_state=seed)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    train_df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return train_df_resampled


def downsample_data(df, thr=500, seed=2023):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()
    
    # identify the classes that have less than the threshold number of samples
    up_classes = class_dist[class_dist > thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    down_dfs = []

    # loop through the undersampled classes and upsample them
    for c in up_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # Remove that class data
        df = df.query("primary_label!=@c")
        # upsample the dataframe
        class_df = class_df.sample(n=thr, replace=False, random_state=seed)
        # append the upsampled dataframe to the list
        down_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    down_df = pd.concat([df] + down_dfs, axis=0, ignore_index=True)
    
    return down_df


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
        cfg.num_classes = len(class_names)
        class_labels = list(range(len(class_names)))
        n2l = dict(zip(class_names, class_labels))
        df_all['label'] = df_all.primary_label.map(lambda x: n2l[x])

    elif cfg.ft_stage == 2:
        class_names = sorted(os.listdir(os.path.join(cfg.birb_23_path, 'train_audio')))
        cfg.num_classes = len(class_names)
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

    # If apply filter
    if cfg.filter_tail:
        df_all = filter_data(df_all, thr=5)
        train_df = df_all.query("fold!=0 | ~cv").reset_index(drop=True)
        val_df = df_all.query("fold==0 & cv").reset_index(drop=True)
    else:
        train_df = df_all.query("fold!=0").reset_index(drop=True)
        val_df = df_all.query("fold==0").reset_index(drop=True)
        
    # # Upsample train data
    # if cfg.upsample_thr:
    #     train_df = upsample_data(train_df, thr=cfg.upsample_thr, seed=cfg.seed)
    # if cfg.downsample_thr:
    #     train_df = downsample_data(train_df, thr=cfg.downsample_thr, seed=cfg.seed)
    if cfg.upsample_thr:
        train_df = upsample_data(train_df, seed=cfg.seed)

    return train_df, val_df


def prepare_data_with_no_labels(cfg):
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
    return train_test_split(df_all, test_size=0.1, random_state=2023)


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
            sr=32000, 
            n_fft=2048,
            n_mels=128, 
            fmin=80, 
            fmax=16000, 
            ref=1.0, 
            amin=1e-10, 
            top_db=None,
            freeze_parameters=True,
        )

    def __call__(self, audio):
        feat = self.spectrogram_extractor(audio.unsqueeze(0))
        feat = self.logmel_extractor(feat)
        feat = feat.squeeze().T
    
        return feat


class BirbDataset(Dataset):
    def __init__(self, cfg, dataframe, augment=True):
        self.df = dataframe
        self.duration = cfg.duration
        self.melbins = cfg.melbins
        self.img_len = cfg.img_len
        self.augment = augment
        
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
                    sample_rate=32000,
                    n_fft=625,
                    f_min=80,
                    f_max=16000,
                ),
                torchaudio.transforms.AmplitudeToDB(),
            )
        elif cfg.feat == 'tl_spec':
            self.spec_transform = TorchLibrosaSpectrogram(cfg)
        
        self.spec_augmenter = None
        if augment:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=64, 
                time_stripes_num=2,
                freq_drop_width=8, 
                freq_stripes_num=2,
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
            # low_freq=80,
            # high_freq=16000,
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
            
        if self.augment:
            feat = feat.unsqueeze(0).unsqueeze(0)
            feat = self.spec_augmenter(feat).squeeze()
        
        return feat, label, audio

    def __len__(self):
        return len(self.df)