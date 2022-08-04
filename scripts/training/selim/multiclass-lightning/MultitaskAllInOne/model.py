import os

# setting tokenizers parallelism to false adds robustness when dploying the model
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# dill import needs to be kept for more robustness in multimodel serialization
# import dill
# dill.extend(True)

from collections import Counter


from tqdm.auto import tqdm

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import numpy as np
from sklearn import metrics

from transformers import AdamW, AutoTokenizer

from torch.optim.lr_scheduler import StepLR

from data import CustomDataset
from utils import flatten, tagname_to_id, compute_weights, beta_score
from architecture import Model

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
        gpus: int,
        n_freezed_layers: int,
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
            output_length,
            n_freezed_layers=n_freezed_layers,
        )
        self.tokenizer = tokenizer
        self.val_params = val_params

        self.training_device = training_device

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
        train_loss = self.Focal_loss(outputs, batch["targets"], weighted=True)

        self.log(
            "train_loss",
            train_loss.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            logger=False,
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        val_loss = self.Focal_loss(outputs, batch["targets"], weighted=False)
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

        scheduler = StepLR(optimizer, step_size=1, gamma=0.6)
        """scheduler = ReduceLROnPlateau(
            optimizer, "min", 0.25, patience=1, threshold=1e-3
        )"""
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

    def hypertune_threshold(self, f_beta: float = 0.8):
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
        optimal_f_beta_scores = {}
        optimal_precision_scores = {}
        optimal_recall_scores = {}

        for j in range(logit_predictions.shape[1]):
            preds_one_column = logit_predictions[:, j]
            min_proba = np.round(min(preds_one_column), 3)
            max_proba = np.round(max(preds_one_column), 3)

            thresholds_list = np.round(np.linspace(max_proba, min_proba, 101), 3)

            f_beta_scores = []
            precision_scores = []
            recall_scoress = []
            for thresh_tmp in thresholds_list:
                score = self.get_metric(
                    preds_one_column,
                    y_true[:, j],
                    f_beta,
                    thresh_tmp,
                )
                f_beta_scores.append(score["f_beta_score"])
                precision_scores.append(score["precision"])
                recall_scoress.append(score["recall"])

            max_threshold = 0
            best_f_beta_score = 0
            best_recall = 0
            best_precision = 0

            for i in range(2, len(f_beta_scores) - 2):

                f_beta_score_mean = np.mean(f_beta_scores[i - 2 : i + 2])
                precision_score_mean = np.mean(precision_scores[i - 2 : i + 2])
                recall_score_mean = np.mean(recall_scoress[i - 2 : i + 2])

                if f_beta_score_mean >= best_f_beta_score:

                    best_f_beta_score = f_beta_score_mean
                    best_recall = recall_score_mean
                    best_precision = precision_score_mean

                    max_threshold = thresholds_list[i]

            tag_name = list(self.tagname_to_tagid.keys())[j]

            optimal_f_beta_scores[tag_name] = best_f_beta_score
            optimal_precision_scores[tag_name] = best_precision
            optimal_recall_scores[tag_name] = best_recall

            optimal_thresholds_dict[tag_name] = max_threshold

        self.optimal_thresholds = optimal_thresholds_dict

        return {
            "precision": optimal_precision_scores,
            "recall": optimal_recall_scores,
            "f_beta_scores": optimal_f_beta_scores,
        }

    def get_metric(self, preds, groundtruth, f_beta, threshold_tmp):
        columns_logits = np.array(preds)
        column_pred = np.array(columns_logits > threshold_tmp).astype(int)

        precision = metrics.precision_score(
            groundtruth,
            column_pred,
            average="binary",
        )
        recall = metrics.recall_score(
            groundtruth,
            column_pred,
            average="binary",
        )
        f_beta_score = beta_score(precision, recall, f_beta)
        return {
            "precision": np.round(precision, 3),
            "recall": np.round(recall, 3),
            "f_beta_score": np.round(f_beta_score, 3),
        }


class FocalLoss(nn.Module):
    def __init__(self, alphas, gamma=0.2):
        super(FocalLoss, self).__init__()
        self.alphas = alphas
        self.gamma = gamma

    def forward(self, outputs, targets, weighted: bool):
        if weighted:
            BCE_loss = F.binary_cross_entropy_with_logits(
                outputs, targets, reduction="mean", pos_weight=self.alphas
            )
        else:
            BCE_loss = F.binary_cross_entropy_with_logits(
                outputs, targets, reduction="mean"
            )
        """pt = torch.exp(-BCE_loss)
        row_loss = ((1 - pt) ** self.gamma) * BCE_loss
        row_mean = torch.mean(row_loss, 0)

        F_loss = torch.dot(row_mean, self.alphas)"""

        return BCE_loss


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
    n_freezed_layers: int,
    weight_decay=0.02,
    warmup_steps=500,
    output_length=384,
    keep_neg_examples_bool=False,
    learning_rate=3e-5,
    training_device: str = "cuda",
    f_beta: float = 0.8,
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
        training_device=training_device,
        keep_neg_examples=keep_neg_examples_bool,
        only_backpropagate_pos=only_backpropagate_pos,
        max_len=max_len,
        n_freezed_layers=n_freezed_layers,
    )

    """lr_finder = trainer.tuner.lr_find(model)
    new_lr = lr_finder.suggestion()
    model.hparams.learning_rate = new_lr"""
    trainer.fit(model)

    model.optimal_scores = model.hypertune_threshold(f_beta)

    del model.training_loader
    del model.val_loader
    # del model.targets

    return model
