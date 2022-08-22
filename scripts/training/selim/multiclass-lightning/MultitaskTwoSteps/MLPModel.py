import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from utils import beta_score


from utils import (
    flatten,
    get_first_level_ids,
)
from loss import FocalLoss


class MLPArchitecture(torch.nn.Module):
    """
    MLP architecture
    """

    def __init__(
        self, ids_each_level, dropout_rate: float, transformer_output_length: int
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation_function = torch.nn.ELU()
        self.normalization = torch.nn.BatchNorm1d(transformer_output_length)

        self.n_tasks = len(ids_each_level)
        self.mid_layer = torch.nn.Linear(
            transformer_output_length * 2 * self.n_tasks, transformer_output_length
        )
        self.output_layer = torch.nn.Linear(
            transformer_output_length, len(flatten(flatten(ids_each_level)))
        )

    def forward(self, input):
        output = self.mid_layer(input)
        output = self.dropout(output)
        output = self.activation_function(output)
        output = self.normalization(output)
        output = self.output_layer(output)
        return output


class TrainingMLP(pl.LightningModule):
    """
    pytorch lightning structure used for trainng the MLP models.
    """

    def __init__(
        self,
        val_params,
        gpus: int,
        tagname_to_tagid,
        loss_alphas,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-7,
        weight_decay: float = 0.1,
        dropout_rate: float = 0.3,
        output_length: int = 384,
        training_device: str = "cuda",
        **kwargs,
    ):

        super().__init__()

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon

        self.tagname_to_tagid = tagname_to_tagid
        self.ids_each_level = get_first_level_ids(self.tagname_to_tagid)
        self.model = MLPArchitecture(
            self.ids_each_level,
            dropout_rate,
            output_length,
        )
        self.val_params = val_params

        self.training_device = training_device

        self.loss_alphas = loss_alphas

        if gpus >= 1:
            self.loss_alphas = self.loss_alphas.to(torch.device("cuda:0"))

        self.Focal_loss = FocalLoss(alphas=self.loss_alphas)
        # self.only_backpropagate_pos = only_backpropagate_pos

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch["X"])
        train_loss = self.Focal_loss(outputs, batch["y"], weighted=True)

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
        outputs = self(batch["X"])
        val_loss = self.Focal_loss(outputs, batch["y"], weighted=False)
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

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.2, patience=5, threshold=1e-3
        )

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def get_loaders(self, dataset, params):

        set = CustomDataset(dataset, self.tagname_to_tagid)
        loader = DataLoader(set, **params, pin_memory=True)
        return loader


class LoggedMLPModel(torch.nn.Module):
    """
    Logged MLPs structure, done for space memory optimization
    Only contains needed varibles and functions for inference
    """

    def __init__(self, trained_model) -> None:
        super().__init__()
        self.trained_architecture = trained_model.model
        self.tagname_to_tagid = trained_model.tagname_to_tagid
        self.val_params = trained_model.val_params

    def forward(self, inputs):
        output = self.trained_architecture(inputs)
        return output

    def get_loaders(self, dataset, params):

        set = CustomDataset(dataset, self.tagname_to_tagid)
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def custom_predict(
        self,
        validation_dataset,
        testing=False,
        hypertuning_threshold: bool = False,
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
        )

        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)

        y_true = []
        logit_predictions = []

        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                total=len(validation_dataset["X"]) // validation_loader.batch_size,
            ):

                if not testing:
                    y_true.append(batch["y"])

                logits = self(batch["X"].to(testing_device)).cpu()
                logit_predictions.append(logits)

        logit_predictions = torch.cat(logit_predictions, dim=0).numpy()
        logit_predictions = sigmoid(logit_predictions)

        target_list = list(self.tagname_to_tagid.keys())
        probabilities_dict = []
        # postprocess predictions
        for i in range(logit_predictions.shape[0]):

            # Return predictions
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
            return logit_predictions, y_true, probabilities_dict

        else:
            return probabilities_dict

    def hypertune_threshold(self, val_data, f_beta: float = 0.8):
        """
        having the probabilities, loop over a list of thresholds to see which one:
        1) yields the best results
        2) without being an aberrant value
        """

        logit_predictions, y_true, _ = self.custom_predict(
            val_data, hypertuning_threshold=True
        )

        optimal_thresholds_dict = {}
        optimal_f_beta_scores = {}
        optimal_precision_scores = {}
        optimal_recall_scores = {}

        for j in range(logit_predictions.shape[1]):
            preds_one_column = logit_predictions[:, j]
            min_proba = np.round(min(preds_one_column), 2)
            max_proba = np.round(max(preds_one_column), 2)

            thresholds_list = np.round(np.linspace(max_proba, min_proba, 101), 2)

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


class CustomDataset(Dataset):
    """MLPs custom dataset"""

    def __init__(self, dataset, tagname_to_tagid):
        self.dataset = dataset
        self.tagname_to_tagid = tagname_to_tagid

    def encode_data(self, index):
        dataset_keys = self.dataset.keys()
        encoded = {"X": self.dataset["X"][index]}

        if "y" in dataset_keys:
            target_indices = [
                self.tagname_to_tagid[target]
                for target in self.dataset["y"][index]
                if target in self.tagname_to_tagid
            ]
            targets = np.zeros(len(self.tagname_to_tagid), dtype=int)
            targets[target_indices] = 1

            encoded["y"] = torch.tensor(targets, dtype=float)

        return encoded

    def __len__(self):
        return len(self.dataset["X"])

    def __getitem__(self, index):
        return self.encode_data(index)
