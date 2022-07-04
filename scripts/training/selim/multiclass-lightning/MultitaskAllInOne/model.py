import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
import dill
from collections import Counter

from sklearn import metrics

dill.extend(True)


from typing import Optional
from tqdm.auto import tqdm

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F


import numpy as np
from sklearn import metrics

# from sklearn.metrics import precision_recall_curve, roc_curve

from transformers import AdamW, AutoTokenizer

from torch.optim.lr_scheduler import ReduceLROnPlateau

from data import CustomDataset
from utils import flatten, tagname_to_id, compute_weights
from architecture import Model
from loss import FocalLoss

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


class Transformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        train_dataset,
        val_dataset,
        train_params,
        val_params,
        tokenizer,
        multiclass,
        gpus: int,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-7,
        warmup_steps: int = 500,
        weight_decay: float = 0.1,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        dropout_rate: float = 0.3,
        max_len: int = 128,
        output_length: int = 384,
        training_device: str = "cuda",
        keep_neg_examples: bool = False,
        only_backpropagate_pos=False,
        **kwargs,
    ):

        super().__init__()
        self.output_length = output_length
        self.save_hyperparameters()
        targets_list = train_dataset["target"].tolist()
        self.tagname_to_tagid = tagname_to_id(targets_list)
        # self.num_labels = len(self.tagname_to_tagid)
        self.get_first_level_ids()
        self.max_len = max_len
        self.model = Model(
            model_name_or_path,
            self.ids_each_level,
            dropout_rate,
            self.output_length,
        )
        self.tokenizer = tokenizer
        self.val_params = val_params

        self.training_device = training_device

        # self.multiclass = multiclass
        self.keep_neg_examples = keep_neg_examples

        self.training_loader = self.get_loaders(
            train_dataset, train_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        self.val_loader = self.get_loaders(
            val_dataset, val_params, self.tagname_to_tagid, self.tokenizer, max_len
        )
        loss_alphas = self.get_loss_alphas(targets_list)

        if gpus >= 1:
            cuda0 = torch.device("cuda:0")
            self.loss_alphas = loss_alphas.to(cuda0)
        else:
            self.loss_alphas = loss_alphas

        self.Focal_loss = FocalLoss(alphas=self.loss_alphas)
        # self.only_backpropagate_pos = only_backpropagate_pos

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        train_loss = self.get_loss(outputs, batch["targets"])

        self.log(
            "train_loss", train_loss.item(), prog_bar=True, on_step=False, on_epoch=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = self.get_loss(outputs, batch["targets"])
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = ReduceLROnPlateau(optimizer, "min", 0.3, patience=1, threshold=1e-3)
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader

    def get_loaders(
        self, dataset, params, tagname_to_tagid, tokenizer, max_len: int = 128
    ):

        set = CustomDataset(dataset, tagname_to_tagid, tokenizer, max_len)
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def get_loss(self, outputs, targets, only_pos: bool = False):
        return self.Focal_loss(outputs, targets)

    def get_loss_alphas(self, targets_list):
        counts = dict(Counter(flatten(targets_list)))
        sorted_counts = [counts[k] for k, v in self.tagname_to_tagid.items()]
        return torch.tensor(
            compute_weights(number_data_classes=sorted_counts, n_tot=len(targets_list)),
            dtype=torch.float64,
        )

    def get_first_level_ids(self):
        all_names = list(self.tagname_to_tagid.keys())
        split_names = [name.split("->") for name in all_names]

        assert np.all([len(name_list) == 3 for name_list in split_names])
        final_ids = []

        tag_id = 0
        first_level_names = list(np.unique([name_list[0] for name_list in split_names]))
        for first_level_name in first_level_names:
            first_level_ids = []
            kept_names = [
                name_list[1:]
                for name_list in split_names
                if name_list[0] == first_level_name
            ]
            second_level_names = list(np.unique([name[0] for name in kept_names]))
            for second_level_name in second_level_names:
                second_level_ids = []
                third_level_names = [
                    name_list[1]
                    for name_list in kept_names
                    if name_list[0] == second_level_name
                ]
                for _ in range(len(third_level_names)):
                    second_level_ids.append(tag_id)
                    tag_id += 1
                first_level_ids.append(second_level_ids)
            final_ids.append(first_level_ids)

        self.ids_each_level = final_ids
        self.flat_ids = [
            small_list for mid_list in final_ids for small_list in mid_list
        ]

    def custom_predict(
        self, validation_dataset, testing=False, hypertuning_threshold: bool = False
    ):
        """
        1) get raw predictions
        2) postprocess them to output an output compatible with what we want in the inference
        """

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if testing:
            self.val_params["num_workers"] = 0

        validation_loader = self.get_loaders(
            validation_dataset,
            self.val_params,
            self.tagname_to_tagid,
            self.tokenizer,
            self.max_len,
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)
        self.eval()
        self.freeze()
        y_true = []
        logit_predictions = []
        indexes = []

        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                total=len(validation_loader.dataset) // validation_loader.batch_size,
            ):

                if not testing:
                    y_true.append(batch["targets"].detach())
                    indexes.append(batch["entry_id"].detach())

                logits = self(
                    {
                        "ids": batch["ids"].to(testing_device),
                        "mask": batch["mask"].to(testing_device),
                        "token_type_ids": batch["token_type_ids"].to(testing_device),
                    }
                )
                # have a matrix like in the beginning
                logits_to_array = np.array([np.array(t) for t in logits.cpu()])
                logit_predictions.append(logits_to_array)

        logit_predictions = np.concatenate(logit_predictions)
        logit_predictions = sigmoid(logit_predictions)

        target_list = list(self.tagname_to_tagid.keys())
        probabilities_dict = []
        # postprocess predictions
        for i in range(logit_predictions.shape[0]):

            # Return predictions
            # row_pred = np.array([0] * self.num_labels)
            row_logits = logit_predictions[i, :]

            # Return probabilities
            probabilities_item_dict = {}
            for j in range(logit_predictions.shape[1]):
                if hypertuning_threshold:
                    probabilities_item_dict[target_list[j]] = row_logits[j]
                else:
                    probabilities_item_dict[target_list[j]] = (
                        row_logits[j] / self.optimal_thresholds[target_list[j]]
                    )

            probabilities_dict.append(probabilities_item_dict)

        if not testing:
            y_true = np.concatenate(y_true)
            indexes = np.concatenate(indexes)
            return indexes, logit_predictions, y_true, probabilities_dict

        else:
            return probabilities_dict

    def hypertune_threshold(self, beta_f1: float = 0.8):
        """
        having the probabilities, loop over a list of thresholds to see which one:
        1) yields the best results
        2) without being an aberrant value
        """

        data_for_threshold_tuning = self.val_loader.dataset.data
        indexes, logit_predictions, y_true, _ = self.custom_predict(
            data_for_threshold_tuning, hypertuning_threshold=True
        )

        optimal_thresholds_dict = {}
        optimal_scores = {}

        for j in range(logit_predictions.shape[1]):
            preds_one_column = logit_predictions[:, j]
            min_proba = np.round(min(preds_one_column), 3)
            max_proba = np.round(max(preds_one_column), 3)
            thresholds_list = np.round(np.linspace(max_proba, min_proba, 101), 3)
            scores = [
                self.get_metric(
                    preds_one_column,
                    y_true[:, j],
                    beta_f1,
                    thresh_tmp,
                )
                for thresh_tmp in thresholds_list
            ]

            max_threshold = 0
            max_score = 0
            for i in range(2, len(scores) - 2):
                score = np.mean(scores[i - 2 : i + 2])
                if score >= max_score:
                    max_score = score
                    max_threshold = thresholds_list[i]

            tag_name = list(self.tagname_to_tagid.keys())[j]

            optimal_scores[tag_name] = max_score
            optimal_thresholds_dict[tag_name] = max_threshold

        self.optimal_thresholds = optimal_thresholds_dict

        return optimal_scores

    def get_metric(self, preds, groundtruth, beta_f1, threshold_tmp):
        columns_logits = np.array(preds)
        column_pred = np.array(columns_logits > threshold_tmp).astype(int)

        metric = metrics.fbeta_score(
            groundtruth,
            column_pred,
            beta_f1,
            average="binary",
        )
        return np.round(metric, 3)


class FocalLoss(nn.Module):
    def __init__(self, alphas, gamma=0.5):
        super(FocalLoss, self).__init__()
        self.alphas = alphas

        self.gamma = gamma

    def forward(self, outputs, targets):
        # self.alphas.to("cuda:0")
        BCE_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        row_loss = ((1 - pt) ** self.gamma) * BCE_loss
        row_mean = torch.mean(row_loss, 0)

        F_loss = torch.dot(row_mean, self.alphas)

        return torch.mean(F_loss)


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
    max_len: int,
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
        max_len=max_len,
    )

    """lr_finder = trainer.tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.hparams.learning_rate = new_lr"""
    trainer.fit(model)

    model.train_f1_score = model.hypertune_threshold(beta_f1)

    del model.training_loader
    del model.val_loader
    # del model.targets

    return model
