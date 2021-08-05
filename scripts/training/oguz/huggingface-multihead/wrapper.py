import sys
import json

sys.path.append(".")

import pandas as pd
import numpy as np
import torch
import mlflow
from torch.utils.data import DataLoader

from data import MultiHeadDataFrame


def extract_predictions(probs, threshold):
    """Extracts predictions from probabilities and threshold"""
    probs = probs > threshold

    preds = []
    for i in range(probs.shape[0]):
        preds.append(np.nonzero(probs[i, :])[0].tolist())
    return preds


def list2str(items, is_int=False):
    """Converts an array of lists of integers into an array of strings"""

    arr = []
    for item in items:
        if len(item) == 0:
            arr.append("")
        elif is_int:
            arr.append(",".join([str(i) for i in item]))
        else:
            arr.append(",".join([f"{i:.3f}" for i in item]))
    return arr


class MLFlowWrapper(mlflow.pyfunc.PythonModel):
    """MLFlow Wrapper class for inference"""

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        # process labels
        with open(context.artifacts["labels"], "r") as f:
            self.labels = [line.strip() for line in f.readlines()]

        # process groups
        if self.model.iterative:
            with open(context.artifacts["groups"], "r") as f:
                self.groups = [line.strip() for line in f.readlines()]

        # process inference params
        with open(context.artifacts["infer_params"], "r") as f:
            self.infer_params = json.load(f)

        # sanity checks for dataset params
        dataset_params = self.infer_params["dataset"]
        assert "filter" not in dataset_params, "Can't use a filter in an inference dataset!"

        if "inference" in dataset_params:
            assert dataset_params["inference"], "Can only use an inference dataset!"
            dataset_params.pop("inference")

    def predict(self, context, model_input):
        # get dataset and data loader
        dataset = MultiHeadDataFrame(
            model_input,
            tokenizer=self.tokenizer,
            filter=None,
            inference=True,
            **self.infer_params["dataset"],
        )
        dataloader = DataLoader(dataset, **self.infer_params["dataloader"])

        # containers for logits
        probs_targets = []
        if self.model.iterative:
            probs_groups = []

        # forward pass
        with torch.no_grad():
            for batch in dataloader:
                for k, v in batch.items():
                    batch[k] = v.to("cuda")

                if self.model.iterative:
                    batch_groups, batch_targets = self.model.forward(
                        batch, group_threshold=self.infer_params["threshold"]["group"]
                    )
                    batch_groups = torch.sigmoid(batch_groups)
                    batch_targets = torch.sigmoid(batch_targets)
                    probs_groups.append(batch_groups.detach().cpu().numpy())
                else:
                    batch_targets = torch.sigmoid(self.model.forward(batch))
                probs_targets.append(batch_targets.detach().cpu().numpy())

        probs_targets = np.concatenate(probs_targets, axis=0)
        preds_targets = extract_predictions(probs_targets, self.infer_params["threshold"]["target"])
        output = {
            "probabilities_target": list2str(probs_targets.tolist()),
            "predictions_target": list2str(preds_targets, is_int=True),
        }

        if self.model.iterative:
            probs_groups = np.concatenate(probs_groups, axis=0)
            preds_groups = extract_predictions(
                probs_groups, self.infer_params["threshold"]["group"]
            )
            output.update(
                {
                    "probabilities_group": list2str(probs_groups.tolist()),
                    "predictions_group": list2str(preds_groups, is_int=True),
                }
            )

        return pd.DataFrame.from_dict(output)
