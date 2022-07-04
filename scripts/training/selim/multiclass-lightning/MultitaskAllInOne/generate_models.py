import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dill import needs to be kept for more robustness in multimodel serialization
import dill

dill.extend(True)


from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import Transformer


def train_model(
    train_dataset,
    val_dataset,
    MODEL_DIR: str,
    MODEL_NAME: str,
    BACKBONE_NAME: str,
    TOKENIZER_NAME: str,
    dropout_rate: float,
    train_params,
    val_params,
    gpu_nb: int,
    MAX_EPOCHS: int,
    weight_decay=0.02,
    warmup_steps=500,
    output_length=384,
    multiclass_bool=True,
    keep_neg_examples_bool=False,
    learning_rate=3e-5,
    training_device: str = "cuda",
    beta_f1: float = 0.8,
    only_backpropagate_pos=False,
):
    PATH_NAME = MODEL_DIR
    if not os.path.exists(PATH_NAME):
        os.makedirs(PATH_NAME)

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")

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
        progress_bar_refresh_rate=40,
        profiler="simple",
        log_gpu_memory=True,
        weights_summary=None,
        gpus=gpu_nb,
        precision=16,
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
        model_name_or_path=BACKBONE_NAME,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_params=train_params,
        val_params=val_params,
        tokenizer=tokenizer,
        gpus=gpu_nb,
        plugin="deepspeed_stage_3_offload",
        accumulate_grad_batches=1,
        max_epochs=MAX_EPOCHS,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        output_length=output_length,
        learning_rate=learning_rate,
        multiclass=multiclass_bool,
        training_device=training_device,
        keep_neg_examples=keep_neg_examples_bool,
        only_backpropagate_pos=only_backpropagate_pos,
        documentation=None,
    )

    """lr_finder = trainer.tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.hparams.learning_rate = new_lr"""
    trainer.fit(model)

    model.train_f1_score = model.hypertune_threshold(beta_f1)

    del model.training_loader
    del model.val_loader
    del model.targets

    return model
