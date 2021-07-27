import argparse
import logging
import os
import sys

import mlflow
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from constants import PILLARS_1D, SUBPILLARS_1D
from data import MultiHeadDataFrame
from model import MultiHeadTransformer
from trainer import MultiHeadTrainer

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
    parser.add_argument(
        "--freeze_backbone",
        dest="freeze_backbone",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: 'ce', 'focal'",
    )
    parser.add_argument("--split", type=str, nargs="+", default="subpillars_1d")
    parser.add_argument("--iterative", action="store_true", default=False)
    parser.add_argument("--save_model", action="store_true", default=False)
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

    # load datasets
    train_df = pd.read_pickle(f"{args.training_dir}/train_df.pickle")
    test_df = pd.read_pickle(f"{args.test_dir}/test_df.pickle")
    logger.info(f" loaded train_dataset length is: {train_df.shape}")
    logger.info(f" loaded test_dataset length is: {test_df.shape}")

    # download model from model hub
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)

    # build classifier model from backbone
    model = MultiHeadTransformer(
        backbone,
        num_heads=len(PILLARS_1D),
        num_classes=[len(group) for group in SUBPILLARS_1D],
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )

    # form datasets out of pandas data frame
    train_dataset = MultiHeadDataFrame(
        train_df,
        tokenizer=tokenizer,
        source="excerpt",
        target="subpillars_1d",
        groups=SUBPILLARS_1D,
        group_names=PILLARS_1D,
        filter=args.split,
        flatten=True,
    )
    test_dataset = MultiHeadDataFrame(
        test_df,
        tokenizer=tokenizer,
        source="excerpt",
        target="subpillars_1d",
        groups=SUBPILLARS_1D,
        group_names=PILLARS_1D,
        filter=args.split,
        flatten=True,
    )

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
    )

    # compute metrics function for binary classification
    def compute_metrics(pred, threshold=0.5):
        labels = pred.label_ids
        preds = pred.predictions > threshold
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    # create trainer instance
    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        focal_loss=(args.loss == "focal"),
    )

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
