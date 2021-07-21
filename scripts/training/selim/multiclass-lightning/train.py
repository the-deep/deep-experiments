import sys
import os
import argparse
import logging
import pickle
from pathlib import Path
import random

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AdamW,
)

from utils import *
from generate_models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--dropout_rate", type=str, default=0.3)
    parser.add_argument("--pred_threshold", type=str, default=0.5)

    parser.add_argument("--model_name", type=str, default='microsoft/xtremedistil-l6-h384-uncased')
    parser.add_argument("--tokenizer_name", type=str, default='microsoft/xtremedistil-l6-h384-uncased')
    #parser.add_argument("--log_every_n_steps", type=int, default=10)
    #parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--method_language", type=str, default='keep all')

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()


    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets

    ########################################

    def read_merge_data (TRAIN_PATH, VAL_PATH, data_format:str='csv'):
        print(f"{TRAIN_PATH}/train.pickle")
    
        if data_format=='pickle':
            train_df = pd.read_pickle(f"{TRAIN_PATH}/train.pickle")
            val_df = pd.read_pickle(f"{VAL_PATH}/val.pickle")
        
        else:
            train_df = pd.read_csv(TRAIN_PATH)
            val_df = pd.read_csv(VAL_PATH)

        all_dataset = pd.concat([train_df, val_df])[['entry_id', 'excerpt', 'subpillars', 'language']]\
                        .rename(columns={'subpillars':'target'})

        # Keep only unique values in pillars
        all_dataset["target"] = all_dataset["target"].apply(lambda x: clean_rows (x))

        # Keep only rows with a not empty pillar
        all_dataset = all_dataset[all_dataset.target.apply(lambda x: len(x)>0)][['entry_id', 'excerpt', 'target', 'language']]
        return all_dataset

    print('importing data ............')
    all_dataset = read_merge_data (args.training_dir, args.val_dir, data_format='pickle')

    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 2
    }

    val_params = {
        'batch_size': args.val_batch_size,
        'shuffle': False,
        'num_workers': 2
        }

    train_df, val_df = preprocess_data(all_dataset, 
                                   perform_augmentation=False,
                                   method='keep en')
    
    tags_ids = tagname_to_id (all_dataset.target)
    list_tags = list(tags_ids.keys())

    number_data_classes = []
    for tag in list_tags:
        nb_data_in_class = train_df.target.apply(lambda x: tag in (x)).sum()
        number_data_classes.append(nb_data_in_class)

    weights = compute_weights (number_data_classes, train_df.shape[0])

    over_sampled_targets = []
    for i in range (len(weights)):
        if weights[i]>5:
            weights[i]=weights[i]**1.5


    log_dir_name = "-".join(args.model_name.split("/"))
    PATH_NAME = log_dir_name + '-subpillars-' + args.method_language
    if not os.path.exists(PATH_NAME):
        os.makedirs(PATH_NAME)
    os.chdir(PATH_NAME)

    early_stopping_callback = EarlyStopping(monitor='val_f1',
                                        patience=2,
                                       mode='max')

    checkpoint_callback_params = {
        'save_top_k': 1,
        'verbose': True,
        'monitor': "val_f1",
        'mode': "max"
    }
    dirpath_pillars = f"./checkpoints-subpillars-{log_dir_name}"
    checkpoint_callback_pillars = ModelCheckpoint(
    dirpath=dirpath_pillars,
    **checkpoint_callback_params
    )

    en_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print('begin training ............')
    en_model_pillars = train_on_specific_targets(train_df,
                                                val_df,
                                                f"subpillars-{log_dir_name}-",
                                                dirpath_pillars,
                                                args.model_name,
                                                en_tokenizer,
                                                early_stopping_callback,
                                                checkpoint_callback_pillars,
                                                gpu_nb=args.n_gpus,
                                                train_params=train_params,
                                                val_params=val_params,
                                                MAX_EPOCHS=args.epochs,
                                                dropout_rate=args.dropout_rate,
                                                weight_classes=weights,
                                                weight_decay=args.weight_decay,
                                                learning_rate=args.learning_rate,
                                                max_len=args.max_len,
                                                warmup_steps=args.warmup_steps,
                                                pred_threshold=args.pred_threshold)

    _ , val_loader = get_loaders (train_df, val_df, train_params, val_params, tags_ids, en_tokenizer)



    metrics = en_model_pillars.custom_eval(val_loader, )
    metrics.loc['mean'] = metrics.mean()

    print (metrics.loc['mean'])
