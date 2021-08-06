import argparse
import logging
import os
import sys

import mlflow
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client
    # parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--model_uri", type=str, required=True)

    # SageMaker parameters - data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
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
    infer_df = pd.read_pickle(f"{args.test_dir}/infer_df.pickle")
    logger.info(f" loaded infer_dataset length is: {infer_df.shape}")
    logger.info(infer_df.head())

    # get model
    pyfunc_wrapper = mlflow.pyfunc.load_model(args.model_uri)

    # set eval batch size
    # python_model = pyfunc_wrapper._model_impl.python_model
    # logging.info(python_model.infer_params)
    # python_model.infer_params["dataloader"]["batch_size"] = args.eval_batch_size

    # predict
    pred_df = pyfunc_wrapper.predict(infer_df)

    # save predictions
    pred_df.to_csv(f"{args.output_data_dir}/preds.csv", header=True, index=True)