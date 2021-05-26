import json
import logging
import sys
import os
from pathlib import Path

import torch
from transformers import (
    Trainer,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerArguments,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    logger.info("model dir is" + str(model_dir))
    logger.info(str(os.listdir(model_dir)))
    logger.info(str(os.listdir(Path(model_dir) / "tokenizer")))
    model_dir = Path(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
    return {"model": model, "tokenizer": tokenizer}


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    data = json.loads(request_body)["inputs"]
    # data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.labels)


# inference
def predict_fn(input_object, model):
    tokenizer, model = model["tokenizer"], model["model"]
    encodings = tokenizer(input_object, truncation=True, padding=True)
    dataset = Dataset(encodings)

    training_args = TrainerArguments(
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=1,
    )

    trainer = Trainer(
        model=model["model"],
        args=training_args,
    )
    predictions = trainer.predict(dataset)
    return predictions


# postprocess
def output_fn(predictions, content_type):
    return json.dumps(predictions)
    # assert content_type == "application/json"
    # res = predictions.cpu().numpy().tolist()
    # return json.dumps(res)
