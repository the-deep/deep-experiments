import argparse
import logging
import os
import sys
import json

import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoModel, AutoTokenizer, TrainingArguments

from constants import SECTORS, PILLARS_1D, SUBPILLARS_1D, PILLARS_2D, SUBPILLARS_2D
from data import MultiHeadDataFrame
from model import MultiHeadTransformer
from wrapper import MLFlowWrapper
from trainer import MultiHeadTrainer
from utils import str2bool, str2list, get_conda_env_specs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--pooling", type=str2bool, default=False)
    parser.add_argument("--freeze_backbone", type=str2bool, default=False)
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal", "focal_star"],
        help="Loss function: 'ce', 'focal', 'focal_star'",
    )
    parser.add_argument(
        "--weighting", type=str, default=None, choices=["inverse", "inverse_square"]
    )
    parser.add_argument(
        "--target",
        type=str2list,
        default="subpillars_1d",
        help="Prediction targets",
    )
    parser.add_argument("--split", type=str2list, default="subpillars_1d")
    parser.add_argument("--iterative", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--model_dim", type=int, default=None, required=False)
    parser.add_argument("--model_name", type=str)

    # MLFlow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--deploy", type=str2bool, default=True)

    # SageMaker parameters - data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    # set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Args: {args}")

    # load datasets
    train_df = pd.read_pickle(f"{args.training_dir}/train_df.pickle")
    test_df = pd.read_pickle(f"{args.test_dir}/test_df.pickle")
    logger.info(f" loaded train_dataset length is: {train_df.shape}")
    logger.info(f" loaded test_dataset length is: {test_df.shape}")

    # download model from model hub
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)

    # get target groups
    groups, group_names = [], []
    for target in args.target:
        if target == "subpillars_1d":
            groups.append(SUBPILLARS_1D)
            group_names.append(PILLARS_1D)
        elif target == "subpillars" or target == "subpillars_2d":
            groups.append(SUBPILLARS_2D)
            group_names.append(PILLARS_2D)
        elif target == "sectors":
            groups.append([SECTORS])
            group_names.append(["Sectors"])
        else:
            raise NotImplementedError

    # sanity check for iterative option
    if args.iterative:
        assert groups is not None, "Provide groups for the 'iterative' option"

    # build classifier model from backbone
    model = MultiHeadTransformer(
        backbone,
        num_classes=[len(group) for group in groups],
        num_layers=args.num_layers,
        dropout=args.dropout,
        pooling=args.pooling,
        freeze_backbone=args.freeze_backbone,
        iterative=args.iterative,
        use_gt_training=True,
        backbone_dim=args.model_dim,
    )

    # form datasets out of pandas data frame
    train_dataset = MultiHeadDataFrame(
        train_df,
        tokenizer=tokenizer,
        source="excerpt",
        targets=args.target,
        groups=groups,
        group_names=group_names,
        exclude=["NOT_MAPPED"],
        filter=args.split,
        flatten=True,
    )
    test_dataset = MultiHeadDataFrame(
        test_df,
        tokenizer=tokenizer,
        source="excerpt",
        targets=args.target,
        groups=groups,
        group_names=group_names,
        exclude=["NOT_MAPPED"],
        filter=None,
        flatten=True,
    )

    # compute metrics function for multi-class classification
    def compute_metrics(pred, threshold=0.5):
        # add prefix to dictionary keys
        def _prefix(dic, prefix):
            return {(prefix + k): v for k, v in dic.items()}

        # compute metrics given preds and labels
        def _compute(preds, labels, average="micro"):
            preds = preds > threshold
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average=average
            )
            accuracy = accuracy_score(labels, preds)
            return {
                "accuracy": accuracy,
                "f1": f1,
                "precision": precision,
                "recall": recall,
            }

        # process pillar texts for MLFlow
        def _process(text):
            text = text.lower()
            text = text.replace(" ", "_")
            text = text.replace(">", "")
            text = text.replace("&", "_")
            return text

        metrics = {}

        if args.iterative:
            # TODO: ensure the ordering is stable
            preds, preds_group = pred.predictions
            labels, labels_group = pred.label_ids

            # group micro evaluation
            metrics.update(_prefix(_compute(preds_group, labels_group, "micro"), "pillar_micro_"))
            # group macro evaluation
            metrics.update(_prefix(_compute(preds_group, labels_group, "macro"), "pillar_macro_"))
            # per group evaluation
            for i, pillar in enumerate(group_names):
                metrics.update(
                    _prefix(
                        _compute(preds_group[:, i], labels_group[:, i], "binary"),
                        f"{_process(pillar)}_binary_",
                    )
                )
        else:
            labels = pred.label_ids
            preds = pred.predictions

        # micro evaluation
        metrics.update(_prefix(_compute(preds, labels, "micro"), "subpillar_micro_"))
        # macro evaluation
        metrics.update(_prefix(_compute(preds, labels, "macro"), "subpillar_macro_"))
        # per head evaluation
        idx = 0
        for i, pillar in enumerate(group_names):
            idx_end = idx + len(groups[i])

            # per head micro evaluation
            metrics.update(
                _prefix(
                    _compute(preds[:, idx:idx_end], labels[:, idx:idx_end], "micro"),
                    f"{_process(pillar)}_micro_",
                )
            )
            # per head macro evaluation
            metrics.update(
                _prefix(
                    _compute(preds[:, idx:idx_end], labels[:, idx:idx_end], "macro"),
                    f"{_process(pillar)}_macro_",
                )
            )

            # per head target evaluation
            for j, subpillar in enumerate(groups[i]):
                metrics.update(
                    _prefix(
                        _compute(preds[:, idx + j], labels[:, idx + j], "binary"),
                        f"{_process(subpillar)}_binary_",
                    )
                )

            # update index
            idx = idx_end

        return metrics

    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=float(args.learning_rate),
        skip_memory_metrics=False,
        label_names=["labels", "groups"] if args.iterative else ["labels"],
        metric_for_best_model="eval_subpillar_micro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        save_total_limit=1,
    )

    # calculate weighting coefficients
    loss_weights, loss_pos_weights = None, None
    if args.weighting == "inverse":
        classes = [target for group in groups for target in group]
        stats = train_dataset.compute_stats()

        loss_weights = [(stats["ALL"] / stats[c]) for c in classes]
        loss_pos_weights = [weight - 1 for weight in loss_weights]
    if args.weighting == "inverse_square":
        raise NotImplementedError

    # create trainer instance
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        loss_fn=args.loss,
        loss_weights=loss_weights,
        loss_pos_weights=loss_pos_weights,
    )

    # set env variable for MLFlow artifact logging
    if args.save_model:
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "TRUE"

    # set remote mlflow server
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    try:
        # train model
        trainer.train()

        # evaluate model
        eval_result = trainer.evaluate(eval_dataset=test_dataset)

        # write eval result to file which can be accessed later in s3 ouput
        logging.info("Logging eval results")
        eval_file = os.path.join(args.output_data_dir, "eval_results.txt")
        with open(eval_file, "w") as writer:
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")
        mlflow.log_artifact(eval_file)

        # write eval result to mlflow
        for key, value in sorted(eval_result.items()):
            mlflow.log_metric(key, value)

        # get labels
        logging.info("Logging model labels")
        if groups is not None:
            labels = [label for gs in groups for label in gs]
        elif args.target == "sectors":
            labels = SECTORS
        else:
            labels = None

        # output labels artifact
        label_file = os.path.join(args.output_data_dir, "labels.txt")
        with open(label_file, "w") as writer:
            for label in labels:
                writer.write(f"{label}\n")
        artifacts = {"labels": label_file}

        if args.iterative:
            # get gorups
            logging.info("Logging model groups")
            labels = [label for label in group_names]

            # output groups artifact
            group_file = os.path.join(args.output_data_dir, "groups.txt")
            with open(group_file, "w") as writer:
                for label in labels:
                    writer.write(f"{label}\n")
            artifacts.update({"groups": group_file})

        # output target names artifact
        logging.info("Logging target names")
        target_file = os.path.join(args.output_data_dir, "targets.txt")
        with open(target_file, "w") as writer:
            writer.write(f"{args.target}\n")
        artifacts.update({"targets": target_file})

        # log experiment params to MLFlow
        mlflow.log_params(vars(args))

        # log tokenizer parameters
        mlflow.log_params(train_dataset.tokenizer_options)

        # set experiment tags
        mlflow.set_tags({"split": args.split, "iterative": args.iterative})

        # output inference artifact
        logging.info("Logging inference params")
        infer_file = os.path.join(args.output_data_dir, "infer_params.json")
        with open(infer_file, "w") as writer:
            json.dump(
                {
                    "dataset": {
                        "source": "excerpt",
                        "target": args.target,
                        "flatten": True,
                    },
                    "dataloader": {
                        "batch_size": args.eval_batch_size,
                        "shuffle": False,
                        "num_workers": 0,
                    },
                    "threshold": {"group": 0.5, "target": 0.5},
                    "output": {"probs_only": True, "flatten": True},
                },
                writer,
            )
            artifacts.update({"infer_params": infer_file})

        if args.deploy:
            # log model with an inference wrapper
            logging.info("Logging deployment model")
            code_path = os.path.abspath(os.path.dirname(__file__))

            mlflow_wrapper = MLFlowWrapper(tokenizer, trainer.model)
            mlflow.pyfunc.log_model(
                python_model=mlflow_wrapper,
                artifact_path="model",
                registered_model_name="multi-head-transformer",
                artifacts=artifacts,
                conda_env=get_conda_env_specs(),
                code_path=os.listdir(code_path),
            )

        # finish mlflow run
        mlflow.end_run()
    except Exception as e:
        logger.exception(e)
        mlflow.end_run("FAILED")

    # save the model to s3
    if args.save_model:
        trainer.save_model(args.model_dir)
