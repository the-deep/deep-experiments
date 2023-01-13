import copy
import torch
from collections import defaultdict
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from typing import Dict, List

from TransformerModel import TrainingTransformer, LoggedTransformerModel


from utils import (
    _create_df_with_translations,
    _create_df_with_chosen_translations,
    get_tagname_to_id,
    get_tags_proportions,
    hypertune_threshold,
    create_train_val_df,
    _undersample_df,
    _get_new_sectors_tags,
    _get_excerpt_without_augmentation,
    _get_n_tokens,
    _flatten,
    _get_new_subsectors_tags,
)


def train_model(
    train_val_dataset: pd.DataFrame,
    MODEL_NAME: str,
    dropout_rate: float,
    train_params,
    val_params,
    gpu_nb: int,
    loss_gamma: float,
    proportions_pow: float,
    hypertune_threshold_bool: bool = True,
    BACKBONE_NAME: str = None,
    TOKENIZER_NAME: str = None,
    MAX_EPOCHS: int = 5,
    max_len: int = 128,
    n_freezed_layers: int = 0,
    weight_decay=0.02,
    output_length=384,
    learning_rate=3e-5,
    training_device: str = "cuda",
    f_beta: float = 0.8,
    delete_long_excerpts: bool = True,
):
    """
    function to train one model
    """
    train_dataset, val_dataset = create_train_val_df(train_val_dataset)

    val_dataset = _create_df_with_translations(val_dataset)

    # train specific preprocessing
    if delete_long_excerpts:
        n_tokens = _get_n_tokens(train_dataset.excerpt.tolist())
        train_dataset = train_dataset.iloc[n_tokens <= int(max_len * 1.5)]

    targets_list = train_dataset["target"].tolist()
    tagname_to_tagid = get_tagname_to_id(targets_list)
    proportions = get_tags_proportions(tagname_to_tagid, targets_list)

    tags_proportions = {
        tagname: proportions[tagid].item()
        for tagname, tagid in tagname_to_tagid.items()
    }

    train_dataset = _create_df_with_chosen_translations(train_dataset)
    train_dataset = _undersample_df(train_dataset, tags_proportions)

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=1 + MAX_EPOCHS // 3, mode="min"
    )

    checkpoint_callback_params = {
        "save_top_k": 1,
        "verbose": True,
        "monitor": "val_loss",
        "mode": "min",
    }

    checkpoint_callback = ModelCheckpoint(
        filename=MODEL_NAME, **checkpoint_callback_params
    )

    trainer = pl.Trainer(
        logger=None,
        callbacks=[early_stopping_callback, checkpoint_callback],
        progress_bar_refresh_rate=20,
        profiler="simple",
        # log_gpu_memory=True,
        weights_summary=None,
        gpus=gpu_nb,
        # precision=16,
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        # strategy="deepspeed_stage_3"
        # overfit_batches=1,
        # limit_predict_batches=2,
        # limit_test_batches=2,
        # fast_dev_run=True,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches: Union[int, float] = 1.0,
    )

    model = TrainingTransformer(
        model_name_or_path=BACKBONE_NAME,
        tokenizer_name_or_path=TOKENIZER_NAME,
        tagname_to_tagid=tagname_to_tagid,
        tags_proportions=proportions,
        loss_gamma=loss_gamma,
        proportions_pow=proportions_pow,
        val_params=val_params,
        plugin="deepspeed_stage_3_offload",
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        output_length=output_length,
        learning_rate=learning_rate,
        training_device=training_device,
        max_len=max_len,
        n_freezed_layers=n_freezed_layers,
    )

    train_dataloader = model.get_loaders(train_dataset, train_params)
    val_dataloader = model.get_loaders(val_dataset, val_params)

    trainer.fit(model, train_dataloader, val_dataloader)
    model.eval()
    model.freeze()

    # create new data structure, containing only needed information for deployment used for logging models
    returned_model = LoggedTransformerModel(model)

    if hypertune_threshold_bool:
        # hypertune decision boundary thresholds for each label
        optimal_thresholds_dict, optimal_scores = hypertune_threshold(
            returned_model, val_dataset, f_beta
        )
        returned_model.optimal_scores = optimal_scores
        returned_model.optimal_thresholds = optimal_thresholds_dict

    # move model to cpu for deployment
    return returned_model


def _relabel_sectors(
    df: pd.DataFrame,
    projects_list_per_tag: Dict[str, List[int]],
    model_args: Dict,
):
    # sectors relabling

    # 2 things to relabel: cross data and non labeled data

    train_val_df = df.copy()

    mask_cross_in_target = train_val_df.target.apply(
        lambda x: "first_level_tags->sectors->Cross" in x
    )
    cross_train_val_df = train_val_df[mask_cross_in_target].copy()
    cross_train_val_df["excerpt"] = cross_train_val_df.apply(
        lambda x: _get_excerpt_without_augmentation(x),
        axis=1,
    )
    non_cross_train_val_df = train_val_df[~mask_cross_in_target].copy()

    sector_projects = {
        k: v
        for k, v in projects_list_per_tag.items()
        if "first_level_tags->sectors->" in k and "Cross" not in k
    }
    projects_all_sectors = set.intersection(*map(set, list(sector_projects.values())))

    non_trained_projects = list(
        set(train_val_df.project_id.tolist()) - projects_all_sectors
    )

    train_val_data_labeled = non_cross_train_val_df[
        non_cross_train_val_df.project_id.isin(projects_all_sectors)
    ]

    model_name = "sectors_relabling"
    transformer_model = train_model(
        MODEL_NAME=model_name,
        train_val_dataset=train_val_data_labeled,
        hypertune_threshold_bool=True,
        f_beta=0.5,
        **model_args,
    )

    train_val_data_relabled = non_cross_train_val_df[
        non_cross_train_val_df.project_id.isin(non_trained_projects)
    ]

    train_val_data_relabled["excerpt"] = train_val_data_relabled.apply(
        lambda x: _get_excerpt_without_augmentation(x),
        axis=1,
    )

    train_val_data_relabled[
        "non_trained_results"
    ] = transformer_model.generate_test_predictions(
        train_val_data_relabled.excerpt, apply_postprocessing=False
    )
    non_trained_project_results = {}  # {prj_id: {entry_id: [tags]}}
    for one_proj_id in non_trained_projects:
        df_one_proj = train_val_data_relabled[
            train_val_data_relabled.project_id == one_proj_id
        ]
        results_one_proj = dict(
            zip(
                df_one_proj.entry_id.tolist(),
                train_val_data_relabled["non_trained_results"],
            )
        )
        non_trained_project_results[one_proj_id] = results_one_proj

    # relabel cross data
    cross_train_val_df["excerpt"] = cross_train_val_df.apply(
        lambda x: _get_excerpt_without_augmentation(x),
        axis=1,
    )
    cross_results_sectors = transformer_model.generate_test_predictions(
        cross_train_val_df.excerpt, apply_postprocessing=False
    )

    # {entry_id: [predictions]}
    cross_results_sectors = dict(
        zip(cross_train_val_df.entry_id.tolist(), cross_results_sectors)
    )

    sector_tags = list(projects_list_per_tag.keys())
    train_val_df["target"] = train_val_df.apply(
        lambda x: _get_new_sectors_tags(
            x,
            cross_results_sectors,
            non_trained_project_results,
            projects_list_per_tag,
            sector_tags,
        ),
        axis=1,
    )

    return train_val_df


def _relabel_subsectors(df: pd.DataFrame, model_args: Dict):

    train_val_df = df.copy()
    tags_list = list(set(_flatten(train_val_df.target)))
    sectors_with_subsectors = list(
        set(
            ["->".join(tag.split("->")[:-1]) for tag in tags_list if "subsector" in tag]
        )
    )

    for one_sector_with_subsectors in sectors_with_subsectors:
        df_one_subsectors = train_val_df.copy()
        sector_name = (
            f"first_level_tags->sectors->{one_sector_with_subsectors.split('->')[1]}"
        )
        df_one_subsectors["target"] = df_one_subsectors["target"].apply(
            lambda x: [tag for tag in x if one_sector_with_subsectors in tag]
        )

        mask_contains_one_subsector = df_one_subsectors.target.apply(
            lambda x: len(x) > 0
        )
        df_one_sector_with_subsectors = df_one_subsectors[mask_contains_one_subsector]
        if len(df_one_sector_with_subsectors) > 50:
            df_one_sector_without_subsectors = df_one_subsectors[
                ~mask_contains_one_subsector
            ]

            df_one_sector_without_subsectors[
                "excerpt"
            ] = df_one_sector_without_subsectors.apply(
                lambda x: _get_excerpt_without_augmentation(x), axis=1
            )

            one_sector_with_subsectors_model = train_model(
                MODEL_NAME=one_sector_with_subsectors,
                train_val_dataset=df_one_sector_with_subsectors,
                hypertune_threshold_bool=True,
                f_beta=0.5,
                **model_args,
            )

            train_val_predictions_one_sector_with_subsectors = (
                one_sector_with_subsectors_model.generate_test_predictions(
                    df_one_sector_without_subsectors.excerpt, apply_postprocessing=False
                )
            )

            # {entry_id: [predictions]}
            train_val_predictions_one_sector_with_subsectors = dict(
                zip(
                    df_one_sector_without_subsectors.entry_id.tolist(),
                    train_val_predictions_one_sector_with_subsectors,
                )
            )

            train_val_df["target"] = train_val_df.apply(
                lambda x: _get_new_subsectors_tags(
                    x,
                    sector_name,
                    train_val_predictions_one_sector_with_subsectors,
                ),
                axis=1,
            )

    return train_val_df
