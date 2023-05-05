import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


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
    

def upsample_data(df, thr=20, seed=2023):
    # get the class distribution
    class_dist = df['primary_label'].value_counts()

    # identify the classes that have less than the threshold number of samples
    down_classes = class_dist[class_dist < thr].index.tolist()

    # create an empty list to store the upsampled dataframes
    up_dfs = []

    # loop through the undersampled classes and upsample them
    for c in down_classes:
        # get the dataframe for the current class
        class_df = df.query("primary_label==@c")
        # find number of samples to add
        num_up = thr - class_df.shape[0]
        # upsample the dataframe
        class_df = class_df.sample(n=num_up, replace=True, random_state=seed)
        # append the upsampled dataframe to the list
        up_dfs.append(class_df)

    # concatenate the upsampled dataframes and the original dataframe
    up_df = pd.concat([df] + up_dfs, axis=0, ignore_index=True)
    
    return up_df


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
        
    # Upsample train data
    if cfg.upsample_thr:
        train_df = upsample_data(train_df, thr=cfg.upsample_thr, seed=cfg.seed)
    if cfg.downsample_thr:
        train_df = downsample_data(train_df, thr=cfg.downsample_thr, seed=cfg.seed)

    return train_df, val_df