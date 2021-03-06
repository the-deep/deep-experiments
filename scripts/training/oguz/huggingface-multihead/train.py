import argparse
import logging
import os
import sys

import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from constants import PILLARS_1D, SUBPILLARS_1D, PILLARS_2D, SUBPILLARS_2D
from data import MultiHeadDataFrame
from model import MultiHeadTransformer
from trainer import MultiHeadTrainer
from utils import str2bool, str2list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
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
        choices=["ce", "focal"],
        help="Loss function: 'ce', 'focal'",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="subpillars_1d",
        choices=[
            "pillars",
            "subpillars",
            "pillars_1d",
            "subpillars_1d",
            "pillars_2d",
            "subpillars_2d",
            "sectors",
        ],
        help="Prediction target",
    )
    parser.add_argument("--split", type=str2list, default="subpillars_1d")
    parser.add_argument("--iterative", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--model_dim", type=int, default=None, required=False)
    parser.add_argument("--model_name", type=str)

    # MLFlow related parameters
    parser.add_argument("--tracking_uri", type=str)
    parser.add_argument("--experiment_name", type=str)

    # SageMaker parameters - data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
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
    if args.target == "subpillars_1d":
        groups = SUBPILLARS_1D
        group_names = PILLARS_1D
    elif args.target == "subpillars" or args.target == "subpillars_2d":
        groups = SUBPILLARS_2D
        group_names = PILLARS_2D
    else:
        groups = None
        group_names = None

    # build classifier model from backbone
    model = MultiHeadTransformer(
        backbone,
        num_heads=len(groups),
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
        target=args.target,
        groups=groups,
        group_names=group_names,
        filter=args.split,
        flatten=True,
    )
    test_dataset = MultiHeadDataFrame(
        test_df,
        tokenizer=tokenizer,
        source="excerpt",
        target=args.target,
        groups=groups,
        group_names=group_names,
        filter=args.split,
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
    )

    # create trainer instance
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        focal_loss=(args.loss == "focal"),
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
        with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
            print("***** Eval results *****")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")

        # write eval result to MLFlow
        for key, value in sorted(eval_result.items()):
            mlflow.log_metric(key, value)

        # log experiment params to MLFlow
        mlflow.log_params(vars(args))
        mlflow.log_params({"groups": groups, "group_names": group_names})

        # set tags
        mlflow.set_tags({"split": args.split, "iterative": args.iterative})

        # finish mlflow run
        mlflow.end_run()
    except Exception as e:
        logger.exception(e)
        mlflow.end_run("FAILED")

    # save the model to s3
    if args.save_model:
        trainer.save_model(args.model_dir)
