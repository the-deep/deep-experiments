import sys
import os
import argparse
import logging
import json

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import timeit

from utils import (
    read_merge_data,
    preprocess_data,
    tagname_to_id,
    compute_weights,
)
from generate_models import train_on_specific_targets

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--pred_threshold", type=float, default=0.4)
    parser.add_argument("--output_length", type=int, default=384)

    parser.add_argument("--model_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased")
    parser.add_argument(
        "--tokenizer_name", type=str, default="microsoft/xtremedistil-l6-h384-uncased"
    )
    # parser.add_argument("--log_every_n_steps", type=int, default=10)
    # parser.add_argument("--n_classes", type=int, default=6)
    parser.add_argument("--method_language", type=str, default="keep all")

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--val_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets

    ########################################

    print("importing data ............")
    all_dataset = read_merge_data(args.training_dir, args.val_dir, data_format="pickle")

    train_params = {"batch_size": args.train_batch_size, "shuffle": True, "num_workers": 4}

    val_params = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": 4}

    train_df, val_df = preprocess_data(
        all_dataset, perform_augmentation=False, method=args.method_language
    )

    tags_ids = tagname_to_id(all_dataset.target)
    list_tags = list(tags_ids.keys())

    number_data_classes = []
    for tag in list_tags:
        nb_data_in_class = train_df.target.apply(lambda x: tag in (x)).sum()
        number_data_classes.append(nb_data_in_class)

    weights = compute_weights(number_data_classes, train_df.shape[0])

    weights = [weight if weight < 5 else weight ** 2 for weight in weights]

    log_dir_name = "-".join(args.model_name.split("/"))
    PATH_NAME = args.model_dir
    if not os.path.exists(PATH_NAME):
        os.makedirs(PATH_NAME)

    early_stopping_callback = EarlyStopping(monitor="val_f1", patience=2, mode="max")

    checkpoint_callback_params = {
        "save_top_k": 1,
        "verbose": True,
        "monitor": "val_f1",
        "mode": "max",
    }
    dirpath_pillars = str(f"{args.model_dir}/checkpoints-subpillars-{log_dir_name}")
    checkpoint_callback_pillars = ModelCheckpoint(filename="model", **checkpoint_callback_params)

    model_subpillars = train_on_specific_targets(
        train_df,
        val_df,
        f"subpillars-{log_dir_name}-",
        dirpath_pillars,
        args.model_name,
        args.tokenizer_name,
        early_stopping_callback,
        checkpoint_callback_pillars,
        gpu_nb=1,
        train_params=train_params,
        val_params=val_params,
        MAX_EPOCHS=args.epochs,
        dropout_rate=args.dropout_rate,
        weight_classes=weights,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        warmup_steps=args.warmup_steps,
        pred_threshold=float(args.pred_threshold),
        output_length=args.output_length,
    )

    start = timeit.default_timer()
    metrics_pillars, metrics_subpillars, ratio_evaluated_sentences = model_subpillars.custom_eval(
        val_df
    )
    stop = timeit.default_timer()

    time_per_hundred_sentences = 100 * (stop - start) / val_df.shape[0]
    general_stats = {
        "time to predicti 100 sentences:": time_per_hundred_sentences,
        "ratio of evaluated sentences": ratio_evaluated_sentences,
    }

    metrics_subpillars.to_csv(f"{args.output_data_dir}/results_subpillars.csv")
    metrics_pillars.to_csv(f"{args.output_data_dir}/results_pillars.csv")
    with open(f"{args.output_data_dir}/general_stats.json", "w") as f:
        json.dump(general_stats, f)
