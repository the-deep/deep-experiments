import json

import numpy as np
import torch
import mlflow
from torch.utils.data import DataLoader

from data import MultiHeadDataFrame


def _extract_predictions(probs, threshold):
    """Extracts predictions from probabilities and threshold"""
    probs = probs > threshold

    preds = []
    for i in range(probs.shape[0]):
        preds.append(np.nonzero(probs[i, :])[0].tolist())
    return preds


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

        # process target name
        with open(context.artifacts["targets"], "r") as f:
            self.targets = [line.strip() for line in f.readlines()]

        # process inference params
        with open(context.artifacts["infer_params"], "r") as f:
            self.infer_params = json.load(f)

        # sanity checks for dataset params
        dataset_params = self.infer_params["dataset"]
        assert "filter" not in dataset_params, "Can't use a filter in an inference dataset!"

        if "inference" in dataset_params:
            assert dataset_params["inference"], "Can only use an inference dataset!"
            dataset_params.pop("inference")

        # sanity check for output params
        output_params = self.infer_params["output"]
        assert (
            output_params["flatten"] and output_params["probs_only"]
        ), "Flattened output is only supported when preds_only is enabled"

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
        preds_targets = _extract_predictions(
            probs_targets, self.infer_params["threshold"]["target"]
        )

        if self.infer_params["output"]["flatten"]:
            # prepare flattened output
            output = [
                {self.labels[j]: probs_targets[i, j] for j in range(probs_targets.shape[1])}
                for i in range(probs_targets.shape[0])
            ]
        else:
            # put probabilities inside a nested dictionary
            output = [
                {
                    "probabilities": {
                        self.targets[0]: {
                            self.labels[j]: probs_targets[i, j]
                            for j in range(probs_targets.shape[1])
                        }
                    }
                }
                for i in range(probs_targets.shape[0])
            ]

            if not self.infer_params["output"]["probs_only"]:
                # append predictions
                output = [
                    out.update({"predictions": {self.targets[0]: preds_targets[i]}})
                    for i, out in enumerate(output)
                ]

        if self.model.iterative:
            probs_groups = np.concatenate(probs_groups, axis=0)
            preds_groups = _extract_predictions(
                probs_groups, self.infer_params["threshold"]["group"]
            )

            if self.infer_params["output"]["flatten"]:
                # append group preds in a flattened format
                output = [
                    out.update(
                        {self.groups[j]: probs_groups[i, j] for j in range(probs_groups.shape[1])}
                    )
                    for i, out in enumerate(output)
                ]
            else:
                # update probabilities field for the group output
                output = [
                    out["probabilities"].update(
                        {
                            self.targets[1]: {
                                self.groups[j]: probs_groups[i, j]
                                for j in range(probs_groups.shape[1])
                            }
                        }
                    )
                    for i, out in enumerate(output)
                ]

            if not self.infer_params["output"]["probs_only"]:
                # append group predictions
                output = [
                    out["predictions"].update({self.targets[1]: preds_groups[i]})
                    for i, out in enumerate(output)
                ]

        return output
