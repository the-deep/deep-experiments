import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils import get_tagname_to_id, get_tags_proportions, hypertune_threshold
from TransformerModel import TrainingTransformer, LoggedTransformerModel


def train_model(
    train_dataset,
    val_dataset,
    MODEL_DIR: str,
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
):
    """
    function to train one model
    """

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

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
        dirpath=MODEL_DIR, filename=MODEL_NAME, **checkpoint_callback_params
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

    targets_list = train_dataset["target"].tolist()

    tagname_to_tagid = get_tagname_to_id(targets_list)
    tags_proportions = get_tags_proportions(tagname_to_tagid, targets_list)

    model = TrainingTransformer(
        model_name_or_path=BACKBONE_NAME,
        tokenizer_name_or_path=TOKENIZER_NAME,
        tagname_to_tagid=tagname_to_tagid,
        tags_proportions=tags_proportions,
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
