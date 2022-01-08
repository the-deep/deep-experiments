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


class CustomTrainer:
    """
    main class used to train model
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        training_column,
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
        keep_neg_examples_bool=False,
        learning_rate=3e-5,
        weighted_loss: str = "sqrt",
        training_device: str = "cuda",
        beta_f1: float = 0.8,
        dim_hidden_layer: int = 256
    ) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_column = training_column
        self.MODEL_DIR = MODEL_DIR
        self.MODEL_NAME = MODEL_NAME
        self.TOKENIZER_NAME = TOKENIZER_NAME
        self.dropout_rate = dropout_rate
        self.train_params = train_params
        self.val_params = val_params
        self.gpu_nb = gpu_nb
        self.MAX_EPOCHS = MAX_EPOCHS
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.output_length = output_length
        self.max_len = max_len
        self.multiclass_bool = multiclass_bool
        self.keep_neg_examples_bool = keep_neg_examples_bool
        self.learning_rate = learning_rate
        self.weighted_loss = weighted_loss
        self.training_device = training_device
        self.beta_f1 = beta_f1
        self.dim_hidden_layer = dim_hidden_layer

    def train_model(self):
        PATH_NAME = self.MODEL_DIR
        if not os.path.exists(PATH_NAME):
            os.makedirs(PATH_NAME)

        early_stopping_callback = EarlyStopping(
            monitor="val_loss", patience=2, mode="min"
        )

        checkpoint_callback_params = {
            "save_top_k": 1,
            "verbose": True,
            "monitor": "val_loss",
            "mode": "min",
        }

        FILENAME = "model_" + self.training_column
        dirpath_pillars = str(PATH_NAME)
        checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath_pillars, filename=FILENAME, **checkpoint_callback_params
        )

        logger = TensorBoardLogger("lightning_logs", name=FILENAME)

        trainer = pl.Trainer(
            logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback],
            progress_bar_refresh_rate=40,
            profiler="simple",
            log_gpu_memory=True,
            weights_summary=None,
            gpus=self.gpu_nb,
            precision=16,
            accumulate_grad_batches=1,
            max_epochs=self.MAX_EPOCHS,
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
        tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_NAME)
        model = Transformer(
            model_name_or_path=self.MODEL_NAME,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            train_params=self.train_params,
            val_params=self.val_params,
            tokenizer=tokenizer,
            column_name=self.training_column,
            gpus=self.gpu_nb,
            plugin="deepspeed_stage_3_offload",
            accumulate_grad_batches=1,
            max_epochs=self.MAX_EPOCHS,
            dropout_rate=self.dropout_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            output_length=self.output_length,
            learning_rate=self.learning_rate,
            multiclass=self.multiclass_bool,
            weighted_loss=self.weighted_loss,
            training_device=self.training_device,
            keep_neg_examples=self.keep_neg_examples_bool,
            dim_hidden_layer=self.dim_hidden_layer
        )

        """lr_finder = trainer.tuner.lr_find(model)
        new_lr = lr_finder.suggestion()
        model.hparams.learning_rate = new_lr"""
        trainer.fit(model)

        model.train_f1_score = model.hypertune_threshold(self.beta_f1)

        del model.training_loader
        del model.val_loader

        return model
