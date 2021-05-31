import json
import logging
import sys
import os
from pathlib import Path
import io

import pandas as pd
import torch
from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    # AutoTokenizer,
    DistilBertTokenizerFast,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings)


def model_fn(model_dir):
    logger.info("model dir is" + str(model_dir))
    logger.info(str(os.listdir(model_dir)))
    logger.info(str(os.listdir(Path(model_dir) / "tokenizer")))
    model_dir = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir / "tokenizer")
    return {"model": model, "tokenizer": tokenizer}


# data preprocessing
def input_fn(request_body, request_content_type):
    if request_content_type != "text/csv":
        raise ValueError("Unsupported content type: {}".format(request_content_type))
    df = pd.read_csv(io.StringIO(request_body))
    data = list(df["inputs"])
    return data


# inference
def predict_fn(input_object, model):
    tokenizer, model = model["tokenizer"], model["model"]
    encodings = tokenizer(input_object, truncation=True, padding=True)
    dataset = Dataset(encodings)

    training_args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        output_dir="tmp_output",
        logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # train_dataset=dataset,
        # eval_dataset=dataset,
    )
    predictions = trainer.predict(dataset)
    return predictions.predictions


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    res = predictions.tolist()
    return json.dumps(res)
