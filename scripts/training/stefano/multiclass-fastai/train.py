import sys
import os
import argparse
import logging
import pickle
from pathlib import Path

import pandas as pd
from fastai.text.all import (
    TextDataLoaders,
    text_classifier_learner,
    RecallMulti,
    PrecisionMulti,
    F1ScoreMulti,
    AWD_LSTM,
    accuracy_multi,
    MultiCategoryBlock,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning_rate", type=str, default=0.02)
    parser.add_argument("--text_col", type=str)
    parser.add_argument("--label_col", type=str)
    parser.add_argument("--multi_category", type=int, default=1)

    # Data, model, and output directories
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--training_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # load datasets
    df = pd.read_pickle(f"{args.training_dir}/df.pickle")
    logger.info(f" loaded train_dataset shape is: {df.shape}")

    if args.multi_category:
        dls = TextDataLoaders.from_df(
            df,
            text_col=args.text_col,
            label_col=args.label_col,
            label_delim=";",
            valid_col="is_valid",
            is_lm=False,  # Mention explicitly that this dataloader is meant for language model
            seq_len=72,  # Pick a sequence length i.e. how many words to feed through the RNN
            bs=args.batch_size,  # Specify the batch size for the dataloader
            y_block=MultiCategoryBlock,
        )
        learn = text_classifier_learner(
            dls,
            AWD_LSTM,
            drop_mult=0.5,
            metrics=[
                accuracy_multi,
                RecallMulti(thresh=0.35),
                PrecisionMulti(thresh=0.35),
                F1ScoreMulti(thresh=0.35),
            ],
        )
        classes = learn.dls.vocab[1]
    else:
        dls = TextDataLoaders.from_df(
            df,
            text_col=args.text_col,
            label_col=args.label_col,
            valid_col="is_valid",
            is_lm=False,  # Mention explicitly that this dataloader is meant for language model
            seq_len=72,  # Pick a sequence length i.e. how many words to feed through the RNN
            bs=args.batch_size,  # Specify the batch size for the dataloader
        )
        learn = text_classifier_learner(
            dls,
            AWD_LSTM,
            drop_mult=0.5,
            metrics=[],
        )
        classes = list(df[args.label_col].value_counts().index)

    learn.fine_tune(int(args.epochs), float(args.learning_rate))

    train_preds, train_targets = learn.get_preds(0)
    test_preds, test_targets = learn.get_preds(1)

    # Output
    with open(Path(args.output_data_dir) / "train_preds.pickle", "wb") as f:
        pickle.dump(train_preds, f)
    with open(Path(args.output_data_dir) / "train_targets.pickle", "wb") as f:
        pickle.dump(train_targets, f)
    with open(Path(args.output_data_dir) / "test_preds.pickle", "wb") as f:
        pickle.dump(test_preds, f)
    with open(Path(args.output_data_dir) / "test_targets.pickle", "wb") as f:
        pickle.dump(test_targets, f)
    with open(Path(args.output_data_dir) / "test_targets.pickle", "wb") as f:
        pickle.dump(test_targets, f)
    with open(Path(args.output_data_dir) / "classes.pickle", "wb") as f:
        pickle.dump(list(classes), f)
