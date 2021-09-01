from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from data_and_model import Transformer

def train_on_specific_targets(
    train_dataset,
    val_dataset,
    training_column:str,
    MODEL_DIR: str,
    MODEL_NAME: str,
    TOKENIZER_NAME: str,
    dropout_rate: float,
    train_params,
    val_params,
    gpu_nb: int,
    MAX_EPOCHS: int,
    weight_decay=0.02,
    warmup_steps=500,
    output_length=384,
    max_len=150,
    multiclass_bool=True,
    learning_rate=3e-5,
    pred_threshold: float = 0.5,
):
    """
    main function used to train model
    """
    # if not os.path.exists(dirpath):
    #    os.makedirs(dirpath)

    # specific_train_dataset = preprocess_df(train_dataset, training_column)
    # specific_val_dataset = preprocess_df(val_dataset, training_column)


    PATH_NAME = MODEL_DIR
    if not os.path.exists(PATH_NAME):
        os.makedirs(PATH_NAME)

    early_stopping_callback = EarlyStopping(monitor="val_f1", patience=2, mode="max")

    checkpoint_callback_params = {
        "save_top_k": 1,
        "verbose": True,
        "monitor": "val_f1",
        "mode": "max",
    }

    FILENAME = "model_" + training_column
    dirpath_pillars = str(PATH_NAME)
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath_pillars, filename=FILENAME, **checkpoint_callback_params
    )

    logger = TensorBoardLogger("lightning_logs", name=FILENAME)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback],
        progress_bar_refresh_rate=30,
        profiler="simple",
        log_gpu_memory=True,
        weights_summary=None,
        gpus=gpu_nb,
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1,
        gradient_clip_algorithm="norm"
        # overfit_batches=1,
        # limit_predict_batches=2,
        # limit_test_batches=2,
        # fast_dev_run=True,
        # limit_train_batches=1,
        # limit_val_batches=1,
        # limit_test_batches: Union[int, float] = 1.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = Transformer(
        model_name_or_path=MODEL_NAME,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_params=train_params,
        val_params=val_params,
        tokenizer=tokenizer,
        column_name=training_column,
        gpus=gpu_nb,
        precision=16,
        plugin="deepspeed_stage_3_offload",
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        output_length=output_length,
        learning_rate=learning_rate,
        pred_threshold=pred_threshold,
        multiclass=multiclass_bool,
    )

    trainer.fit(model)

    return model
