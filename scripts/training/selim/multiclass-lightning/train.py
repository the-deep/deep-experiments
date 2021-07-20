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
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=str, default=3e-5)
    parser.add_argument("--dropout_rate", type=str, default=0.3)

    parser.add_argument("--model_name", type=str, default='microsoft/xtremedistil-l6-h384-uncased')
    parser.add_argument("--tokenizer_name", type=str, default='microsoft/xtremedistil-l6-h384-uncased')
    #parser.add_argument("--log_every_n_steps", type=int, default=10)
    #parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--method_language", type=str, default='keep en')

    # Data, model, and output directories
    #parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    #parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    #parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument("--training_dir",
                        type=str,
                        default=os.path.join("..", "..", "..", "..", "data", "frameworks_data", "data_v0.4.4", "data_v0.4.4_train.csv"))
    parser.add_argument("--val_dir",
                        type=str,
                        default=os.path.join("..", "..", "..", "..", "data", "frameworks_data", "data_v0.4.4", "data_v0.4.4_val.csv"))
    args, _ = parser.parse_known_args()

    GPU_NB=1

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets

    ########################################
    all_dataset = preprocess_data (args.training_dir, args.val_dir)

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


    #Each pillar alone: used to compute weighted loss and later for each subpillar model
    en_capacities_response_train_dataset, en_capacities_response_val_dataset =\
                get_subpillar_datasets ('Capacities & Response', 
                                        all_dataset,
                                        perform_augmentation=False,
                                        method=args.method_language)

    en_hum_conditions_train_dataset, en_hum_conditions_val_dataset =\
                    get_subpillar_datasets ('Humanitarian Conditions', 
                                            all_dataset,
                                            perform_augmentation=False,
                                            method=args.method_language)

    en_impact_train_dataset, en_impact_val_dataset =\
                                            get_subpillar_datasets ('Impact', 
                                        all_dataset,
                                        perform_augmentation=False,
                                            method=args.method_language)

    en_people_at_risk_train_dataset, en_people_at_risk_val_dataset =\
                                            get_subpillar_datasets ('People At Risk',
                                        all_dataset,
                                        perform_augmentation=False,
                                            method=args.method_language)

    en_priority_interventions_train_dataset, en_priority_interventions_val_dataset = \
                    get_subpillar_datasets ('Priority Interventions', 
                                            all_dataset,
                                            perform_augmentation=False,
                                            method=args.method_language)

    en_priority_needs_train_dataset, en_priority_needs_val_dataset =\
                                            get_subpillar_datasets ('Priority Needs', 
                                        all_dataset, 
                                        perform_augmentation=False,
                                            method=args.method_language)

    en_tot_train = pd.concat([en_capacities_response_train_dataset,
                        en_hum_conditions_train_dataset,
                        en_impact_train_dataset,
                        en_people_at_risk_train_dataset,
                        en_priority_interventions_train_dataset,
                        en_priority_needs_train_dataset])[['entry_id', 'excerpt', 'pillars']]\
                    .rename(columns={'pillars': 'target'})

    en_tot_val = pd.concat([en_capacities_response_val_dataset,
                        en_hum_conditions_val_dataset,
                        en_impact_val_dataset,
                        en_people_at_risk_val_dataset,
                        en_priority_interventions_val_dataset,
                        en_priority_needs_val_dataset])[['entry_id', 'excerpt', 'pillars']]\
                    .rename(columns={'pillars': 'target'})

    en_number_data_classes = [en_capacities_response_train_dataset.shape[0],
                            en_hum_conditions_train_dataset.shape[0],
                            en_impact_train_dataset.shape[0],
                            en_people_at_risk_train_dataset.shape[0],
                            en_priority_interventions_train_dataset.shape[0],
                            en_priority_needs_train_dataset.shape[0]]

    en_pillars_weights = compute_weights (en_number_data_classes, en_tot_train.shape[0])

    log_dir_name = "-".join(args.model_name.split("/"))
    PATH_NAME = log_dir_name + '-pillars-' + args.method_language
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
    dirpath_pillars = f"./checkpoints-pillars-{log_dir_name}"
    checkpoint_callback_pillars = ModelCheckpoint(
    dirpath=dirpath_pillars,
    **checkpoint_callback_params
    )

    en_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    en_model_pillars = train_on_specific_targets(en_tot_train,
                                                en_tot_val,
                                                f"pillars-{log_dir_name}-",
                                                dirpath_pillars,
                                                args.model_name,
                                                en_tokenizer,
                                                early_stopping_callback,
                                                checkpoint_callback_pillars,
                                                gpu_nb=GPU_NB,
                                                train_params=train_params,
                                                val_params=val_params,
                                                MAX_EPOCHS=args.epochs,
                                                dropout_rate=args.dropout_rate,
                                                weight_classes=en_pillars_weights,
                                                weight_decay=args.weight_decay,
                                                learning_rate=args.learning_rate,
                                                max_len=args.max_len,
                                                warmup_steps=args.warmup_steps)

    tagname_to_tagid = tagname_to_id (en_tot_train["target"])
    _ , val_loader = get_loaders (en_tot_train, en_tot_val, train_params, val_params, tagname_to_tagid, en_tokenizer)
    metrics = en_model_pillars.custom_eval(val_loader, )
    metrics.loc['mean'] = metrics.mean()

    metrics.to_csv('metrics.csv')
