import sys
import json

sys.path.append(".")

import pandas as pd
import numpy as np
import torch
import mlflow
from torch.utils.data import DataLoader

from data import MultiHeadDataFrame


def extract_predictions(logits, threshold):
    logits = logits > threshold

    preds = []
    for i in range(logits.shape[0]):
        preds.append(logits[i, np.nonzero(logits[i, :])].tolist())
    return preds


class MLFlowWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model.eval()
        super().__init__()

    def load_context(self, context):
        # process inference params
        with open(context["infer_params"], "r") as f:
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
        logits_targets = []
        if self.model.iterative:
            logits_groups = []

        # forward pass
        with torch.no_grad():
            for batch in dataloader:
                if self.model.iterative:
                    batch_groups, batch_targets = self.model.forward(
                        batch, group_threshold=self.infer_params["threshold"]["group"]
                    )
                    logits_groups.append(batch_groups.detach().numpy())
                else:
                    batch_targets = self.model.forward(batch)
                logits_targets.append(batch_targets.detach().numpy())

        logits_targets = np.concatenate(logits_targets, axis=0)
        preds_targets = extract_predictions(
            logits_targets, self.infer_params["threshold"]["target"]
        )
        output = {
            "logits": [
                ",".join(f"{score:.3f}")
                for i in range(logits_targets.shape[0])
                for score in logits_targets[i, :].tolist()
            ],
            "predictions": [",".join(preds) for preds in preds_targets],
        }

        if self.model.iterative:
            logits_groups = np.concatenate(logits_groups, axis=0)
            preds_groups = extract_predictions(
                logits_groups, self.infer_params["threshold"]["group"]
            )
            output.extend(
                {
                    "logits_group": [
                        ",".join(f"{score:.3f}")
                        for i in range(logits_groups.shape[0])
                        for score in logits_groups[i, :].tolist()
                    ],
                    "predictions_group": [",".join(preds) for preds in preds_groups],
                }
            )

        return pd.DataFrame.from_dict(output)
