import pytorch_lightning as pl
from torch import nn
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from data import ExtractionDataset


class EntryExtractor(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        slice_length: int,
        n_freezed_layers: int = 1,
    ):

        super().__init__()
        self.common_backbone = AutoModel.from_pretrained(model_name_or_path)
        self.common_backbone.encoder.layer = self.common_backbone.encoder.layer[:-1]
        self.slice_length = slice_length
        self.num_labels = num_labels

        # freeze embeddings
        for param in self.common_backbone.embeddings.parameters():
            param.requires_grad = False
        # freeze n_freezed_layers first layers
        if n_freezed_layers > 0:
            for layer in self.common_backbone.encoder.layer[:n_freezed_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.separate_layers = torch.nn.ModuleList(
            [
                AutoModel.from_pretrained(model_name_or_path).encoder.layer[-1]
                for _ in range(num_labels)
            ]
        )

    def forward(self, input_ids, attention_mask):
        logits = []

        hidden_state = self.common_backbone(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

        for separate_layer_idx in range(self.num_labels):
            h = self.separate_layers[separate_layer_idx](hidden_state.clone())[0]

            cls_output = h[..., 0]
            mean_pooling = torch.mean(h, dim=-1)

            output = (cls_output + mean_pooling) / 2

            logits.append(
                output  
            )  
        return torch.stack(logits, dim=2)


class TrainingExtractionModel(pl.LightningModule):
    def __init__(
        self,
        backbone,
        num_labels,
        slice_length,
        extra_context_length,
        lr: float = 1e-4,
        adam_epsilon: float = 1e-7,
        weight_decay: float = 1e-2,
    ):
        """
        Args:
            backbone: a string indicating the backbone model to use
            tokenizer: the used tokenizer
            num_labels: number of labels
            token_loss_weight: weight of the token-level loss e.g. 0.5 will
                result in even weighting of token-level and sentence-level loss
            slice_length: length of the context that is fed into the model at
                once
            extra_context_length: length of prefix that will be fed to the
                model as additional context (without generating predictions)
            n_separate_layers: number of separate layers to use for different
                `separate_layer_groups`
            separate_layer_groups: list of lists of label indices indicating
                how to group labels into separate final layers
        """
        super().__init__()

        self.entry_extraction_model = EntryExtractor(backbone, num_labels, slice_length)

        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.num_labels = num_labels
        self.slice_length = slice_length
        self.extra_context_length = extra_context_length
        self.lr = lr
        self.adam_epsilon = adam_epsilon
        self.weight_decay = weight_decay

    def _compute_loss(self, logits, groundtruth):

        token_loss = F.binary_cross_entropy_with_logits(
            logits, groundtruth.float(), reduction="mean"
        )

        return token_loss

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
    ):
        output = self.entry_extraction_model(input_ids, attention_mask)
        output = output[torch.where(loss_mask == 1)]
        return output

    def _operate_train_or_val_step(self, batch):
        """
        batch: {
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "token_labels": d["token_labels"],
            "loss_mask": d["loss_mask"]
        }
        """

        logits = self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["loss_mask"],
        )

        mask = batch["loss_mask"]
        important_labels = batch["token_labels"]
        important_labels = important_labels[torch.where(mask == 1)]

        loss = self._compute_loss(logits, important_labels)

        return loss

    def training_step(self, batch, batch_idx):

        train_loss = self._operate_train_or_val_step(batch)

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
        val_loss = self._operate_train_or_val_step(batch)

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=False,
        )

        return {"val_loss": val_loss}

    def configure_optimizers(self, *args, **kwargs):
        "Prepare optimizer and schedule (linear warmup and decay)"

        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.adam_epsilon,
        )

        # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        scheduler = ReduceLROnPlateau(
            optimizer, "min", factor=0.5, patience=1, threshold=1e-3
        )

        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]

    def _get_loaders(self, data, params, training_mode: bool):
        """
        get the dataloader from raw data
        """
        set = ExtractionDataset(
            dset=data,  # dict if training and raw text input text if test
            training_mode=training_mode,
            tokenizer=self.tokenizer,
            max_input_len=self.slice_length,
            extra_context_length=self.extra_context_length,
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def hypertune_threshold(self, validation_loader):
        """
        After training the model, generate predictions on validation set and get the optimal decision threshold for
        each label, maximizing a chosen parameter (eg. fscore, recall, precision ...)
        """
        pass


class LoggedExtractionModel(nn.Module):
    def __init__(self, trained_model) -> None:
        super().__init__()

        # get all values needed for inference into new class
        # new class used for logging
        self.trained_entry_extraction_model = trained_model.entry_extraction_model
        self.tokenizer = trained_model.tokenizer
        self.slice_length = trained_model.slice_length
        self.extra_context_length = trained_model.extra_context_length
        self.num_labels = trained_model.num_labels

        self.test_params = {"batch_size": 16, "shuffle": False, "num_workers": 0}

    def forward(
        self,
        input_ids,
        attention_mask,
        loss_mask,
    ):
        output = self.entry_extraction_model(input_ids, attention_mask)
        output = output[torch.where(loss_mask == 1)]
        return output

    def _get_loaders(self, data, params, training_mode: bool):
        """
        get the dataloader from raw data

        returned batch: {
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "loss_mask": d["loss_mask"]
        }
        """
        set = ExtractionDataset(
            dset=data,  # dict if training and raw text input text if test
            training_mode=training_mode,
            tokenizer=self.tokenizer,
            max_input_len=self.slice_length,
            extra_context_length=self.extra_context_length,
        )
        loader = DataLoader(set, **params, pin_memory=True)
        return loader

    def get_highlights(self, raw_input_text):
        """
        only function called in inference
        return predictions from raw input text
        """
        test_loader = self.get_loaders(
            raw_input_text,
            self.val_params,
        )

        # can be on cpu
        if torch.cuda.is_available():
            testing_device = "cuda"
        else:
            testing_device = "cpu"

        self.to(testing_device)

        with torch.no_grad():
            ...
