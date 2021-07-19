import os

from typing import Optional

from icecream import ic
from tqdm.auto import tqdm

import torchmetrics
from torchmetrics.functional import accuracy, f1, auroc

import pytorch_lightning as pl
from pytorch_lightning.core.decorators import auto_move_data

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoTokenizer,
)
from transformers.optimization import (
    Adafactor,
    get_linear_schedule_with_warmup,
)

import tensorflow as tf
import re

class CustomDataset(Dataset):
    def __init__(self, dataframe, tagname_to_tagid, tokenizer, max_len:int=200):
        self.tokenizer = tokenizer
        self.data = dataframe

        self.excerpt_text = dataframe["excerpt"].tolist(
        ) if dataframe is not None else None

        self.targets = self.data['target'].tolist(
        ) if dataframe is not None else None

        self.entry_ids = self.data['entry_id'].tolist(
        ) if dataframe is not None else None

        self.tagname_to_tagid = tagname_to_tagid
        self.tagid_to_tagname = list(tagname_to_tagid.keys())
        self.max_len = max_len

    def encode_example(self,
                       excerpt_text: str,
                       index=None,
                       as_batch: bool = False):
        
        inputs = self.tokenizer(excerpt_text,
                                None,
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding="max_length",
                                return_token_type_ids=True)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        
        targets = None
        if self.targets:
            target_indices = [
                self.tagname_to_tagid[target]
                for target in self.targets[index]
                if target in self.tagname_to_tagid
            ]
            targets = np.zeros(len(self.tagname_to_tagid), dtype=np.int)
            targets[target_indices] = 1
        encoded = {
            'ids':
            torch.tensor(ids, dtype=torch.long),
            'mask':
            torch.tensor(mask, dtype=torch.long),
            'token_type_ids':
            torch.tensor(token_type_ids, dtype=torch.long),
            'targets':
            torch.tensor(targets, dtype=torch.float32)
            if targets is not None else None,
            'entry_id':
            self.entry_ids[index]
        }

        if as_batch:
            return {
                "ids": encoded["ids"].unsqueeze(0),
                "mask": encoded["mask"].unsqueeze(0),
                "token_type_ids": encoded["ids"].unsqueeze(0)
            }
        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)


class Model(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels:int, dropout_rate=0.3, output_length=384):
        super().__init__()
        self.l1 = AutoModel.from_pretrained(model_name_or_path)
        self.l2 = torch.nn.Dropout(dropout_rate)
        self.l3 = torch.nn.Linear(output_length, num_labels)
        
    def forward(self, inputs):
        output = self.l1(inputs["ids"],
                        attention_mask=inputs["mask"],)
        output = output.last_hidden_state
        output = self.l2(output)
        output = self.l3(output)
        return output[:, 0, :]


class Transformer(pl.LightningModule):
    def __init__(self,
                 model_name_or_path: str,
                 num_labels: int,
                 empty_dataset: CustomDataset,
                 training_loader,
                 val_loader,      
                 weight_classes,  
                 pred_threshold: float = .5,
                 learning_rate: float = 1e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 eval_splits: Optional[list] = None,
                 dropout_rate: float = 0.3,
                 output_length=384,

                 **kwargs):
        super().__init__()
        self.output_length = output_length
        self.save_hyperparameters()
        self.num_labels = num_labels
        self.model = Model(model_name_or_path, num_labels, dropout_rate, self.output_length)
        if any(weight_classes):
            self.use_weights = True
            self.weight_classes = torch.tensor(weight_classes).to('cuda')
        else:
            self.use_weights = False
        self.empty_dataset = empty_dataset
        self.pred_threshold = pred_threshold
        self.val_loader = val_loader
        self.training_loader = training_loader

        self.f1_score_train = torchmetrics.F1(
            num_classes=2,
            threshold=0.5,
            average='macro',
            mdmc_average="samplewise",
            ignore_index=None,
            top_k=None,
            multiclass=True,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )

        self.f1_score_val = torchmetrics.F1(
            num_classes=2,
            threshold=0.5,
            average='macro',
            mdmc_average="samplewise",
            ignore_index=None,
            top_k=None,
            multiclass=True,
            compute_on_step=True,
            dist_sync_on_step=False,
            process_group=None,
            dist_sync_fn=None,
        )
        
    @auto_move_data
    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        if self.use_weights:
            loss = F.binary_cross_entropy_with_logits(outputs,
                                                    batch["targets"],
                                                    weight=self.weight_classes)
        else:
            loss = F.binary_cross_entropy_with_logits(outputs,
                                                    batch["targets"])

        self.f1_score_train(torch.sigmoid(outputs),
                            batch["targets"].to(dtype=torch.long))
        self.log("train_f1", self.f1_score_train, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(batch)
        val_loss = F.binary_cross_entropy_with_logits(outputs,
                                                      batch["targets"])

        self.f1_score_val(torch.sigmoid(outputs),
                          batch["targets"].to(dtype=torch.long))
        self.log("val_f1",
                 self.f1_score_val,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        
        self.log("val_loss",
                 val_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return {'val_loss': val_loss, 'val_f1': self.f1_score_val}

    def test_step(self, batch, batch_nb):
        logits = self(batch)
        preds = (torch.sigmoid(logits) > .5)
        return {"preds": preds, "targets_i": batch["targets"]}

    def on_test_epoch_end(self, outputs):
        preds = torch.cat([output["preds"] for output in outputs]).cpu()
        targets = torch.cat([output["targets_i"] for output in outputs]).cpu()
        recalls = []
        precisions = []
        f1_scores = []
        for i in range(targets.shape[1]):
            class_roc_auc = auroc(preds[:, i], targets[:, i])
            self.log(
                f"{self.empty_dataset.sectorid_to_sectorname[i]}_roc_auc/Train",
                class_roc_auc)
            class_f1 = metrics.f1_score(targets[:, i], preds[:, i])
            self.log(
                f"{self.empty_dataset.sectorid_to_sectorname[i]}_f1/Train",
                class_f1)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        output = self(batch)
        return {"logits": output}

    def on_predict_epoch_end(self, outputs):
        logits = torch.cat([output["logits"] for output in outputs[0]])
        preds = torch.sigmoid(logits) >= self.pred_threshold
        pred_classes = []
        for pred in preds:
            pred_classes_i = [
                self.empty_dataset.sectorid_to_sectorname[i]
                for i, p in enumerate(pred) if p
            ]
            pred_classes.append(pred_classes_i)
        self.log({"pred_classes": pred_classes})

    def custom_predict(self, validation_loader, name:str, return_logits=False):
        if self.device.type == "cpu":
            self.to("cuda")
        self.eval()
        self.freeze()
        indexes=torch.tensor([])

        with torch.no_grad():
            iter=0
            for batch in tqdm(validation_loader, total=len(validation_loader.dataset)//validation_loader.batch_size):
                
                logits = self({"ids": batch["ids"].to('cuda'),
                                "mask": batch["mask"].to('cuda'),
                                "token_type_ids": batch["token_type_ids"].to('cuda')})
                logits_to_array = np.array([np.array(t) for t in logits.cpu()])
                
                if return_logits:
                    if iter==0:
                        
                        predictions = logits_to_array
                        indexes = batch["entry_id"]
                    
                    else:
                        predictions = np.concatenate([predictions, logits_to_array], 0) #.append(preds_batch)
                        indexes = tf.concat([indexes, batch["entry_id"]], 0)
                
                    iter += 1

                else:
                    preds_batch = np.zeros(logits.shape, dtype=np.int)
                    preds_batch[(torch.sigmoid(logits) >= self.pred_threshold).cpu().nonzero(as_tuple=True)] = 1
                    if iter==0:
                        predictions = preds_batch
                        indexes = batch["entry_id"]
                    
                    else:
                        predictions = np.concatenate([predictions,preds_batch], 0) #.append(preds_batch)
                        indexes = tf.concat([indexes, batch["entry_id"]], 0)
                
                    iter += 1
                
        np.save('predictions-'+name, np.array(predictions))
        np.save('indexes-'+name, np.array(indexes))
        return np.array(predictions), np.array(indexes)

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        self.dataset_size = len(self.train_dataloader().dataset)
        num_devices = max(1, self.hparams.gpus)  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * num_devices
        return (self.dataset_size /
                effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps())
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.val_loader
    
    def custom_eval(self, eval_dataloader):
        if self.device.type == "cpu":
            self.to("cuda")
        self.eval()
        self.freeze()
        preds_val_all = []
        y_true = []

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, total=len(eval_dataloader.dataset)//eval_dataloader.batch_size):
                logits = self({"ids": batch["ids"].to("cuda"), "mask": batch["mask"].to("cuda"), "token_type_ids": batch["token_type_ids"].to("cuda")})
                preds_batch = np.zeros(logits.shape, dtype=np.int)
                preds_batch[(torch.sigmoid(logits) > self.pred_threshold).cpu().nonzero(as_tuple=True)] = 1
                preds_val_all.append(preds_batch)
                y_true.append(batch["targets"].numpy().astype(np.int))

        preds_val_all = np.concatenate(preds_val_all)
        y_true = np.concatenate(y_true)

        f1_scores = []
        recalls = []
        precisions = []
        accuracies = []
        supports = []
        tagname_to_tagid = self.empty_dataset.tagname_to_tagid
        for tag_name, tag_id in tagname_to_tagid.items():
            cls_rprt = classification_report(y_true[:, tag_id], preds_val_all[:, tag_id], output_dict=True)
            precisions.append(cls_rprt["macro avg"]["precision"])
            recalls.append(cls_rprt["macro avg"]["recall"])
            f1_scores.append(cls_rprt["macro avg"]["f1-score"])
            accuracies.append(cls_rprt["accuracy"])

        metrics_df = pd.DataFrame({
            "Sector": list(tagname_to_tagid.keys()),
            "Precision": precisions,
            "Recall": recalls,
            "F1 Score": f1_scores,
            "Accuracy": accuracies,
        })
        return metrics_df
