from typing import List

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

"""Evaluation logic here only work with flattened datasets."""


def _prefix(dic, prefix):
    """Adds prefix to dictionary keys"""

    return {(prefix + k): v for k, v in dic.items()}


def _process(text):
    """Replaces special characters in text (for MLFlow)"""
    text = text.lower()
    text = text.replace(" ", "_")
    text = text.replace("&", "_")
    text = text.replace(">", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    return text


# compute metrics given preds and labels
def _compute(preds, labels, average="micro", threshold=0.5):
    preds = preds > threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    accuracy = accuracy_score(labels, preds)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def compute_multiclass_metrics(preds, labels, names: List[str], threshold: float = 0.5):
    """Compute metrics for multi-target classification tasks"""
    metrics = {}

    # micro evaluation
    metrics.update(_prefix(_compute(preds, labels, "micro"), "micro_"), threshold=threshold)
    # macro evaluation
    metrics.update(_prefix(_compute(preds, labels, "macro"), "macro_"), threshold=threshold)

    # per class evaluation
    for idx, name in enumerate(names):
        # per class micro evaluation
        metrics.update(
            _prefix(
                _compute(
                    preds[:, idx : idx + 1], labels[:, idx : idx + 1], "binary", threshold=threshold
                ),
                f"{_process(name)}_binary_",
            )
        )

    return metrics


def compute_multitarget_metrics(
    preds,
    labels,
    groups: List[List[str]],
    group_names: List[str],
    threshold: float = 0.5,
):
    metrics = {}
    start = 0
    for idx, group_name in enumerate(group_names):
        metrics.update(
            _prefix(
                compute_multiclass_metrics(
                    preds[:, start : start + len(groups[idx])],
                    labels[:, start : start + len(groups[idx])],
                    names=groups[idx],
                    threshold=threshold,
                ),
                f"{_process(group_name)}_",
            )
        )
        start = start + len(groups[idx])

    # micro evaluation
    metrics.update(_prefix(_compute(preds, labels, "micro", threshold=threshold), "micro_"))
    # macro evaluation
    metrics.update(_prefix(_compute(preds, labels, "macro", threshold=threshold), "macro_"))

    return metrics


def compute_multihead_metrics(
    preds,
    labels,
    groups: List[List[List[str]]],
    group_names: List[List[str]],
    targets: List[str],
    threshold: float = 0.5,
):
    metrics = {}
    for idx, target in enumerate(targets):
        metrics.update(
            _prefix(
                compute_multitarget_metrics(
                    preds[idx],
                    labels[idx],
                    groups=groups[idx],
                    group_names=group_names[idx],
                    threshold=threshold,
                ),
                f"{_process(target)}_",
            )
        )

    return metrics
