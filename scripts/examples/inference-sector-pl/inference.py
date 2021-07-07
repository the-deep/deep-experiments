import json
import logging
import sys

import mlflow
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


MODEL_NAME = "sentence-transformers/paraphrase-mpnet-base-v2"


class SectorsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.excerpt_text = dataframe["excerpt"].tolist() if dataframe is not None else None
        self.max_len = max_len

    def encode_example(self, excerpt_text: str, index=None, as_batch: bool = False):
        inputs = self.tokenizer(
            excerpt_text,
            None,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        encoded = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }
        if as_batch:
            return {
                "ids": encoded["ids"].unsqueeze(0),
                "mask": encoded["mask"].unsqueeze(0),
                "token_type_ids": encoded["ids"].unsqueeze(0),
            }
        return encoded

    def __len__(self):
        return len(self.excerpt_text)

    def __getitem__(self, index):
        excerpt_text = str(self.excerpt_text[index])
        return self.encode_example(excerpt_text, index)


def model_fn(model_data):
    logger.info("model dir is" + str(model_data))
    model = mlflow.pytorch.load_model(model_data, map_location=torch.device("cpu"))
    model.cpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return {"model": model, "tokenizer": tokenizer}


# data preprocessing
def input_fn(request_body, request_content_type):
    assert request_content_type == "application/json"
    logging.info("request_body: {}".format(request_body))
    data = json.loads(request_body)
    logging.info("data: {}".format(data))
    # data = torch.tensor(data, dtype=torch.float32, device=device)
    return data


# inference
def predict_fn(input_object, model):
    logging.info("input_object: {}".format(input_object))
    tokenizer, model = model["tokenizer"], model["model"]
    data = pd.DataFrame.from_dict(input_object)
    dataset = SectorsDataset(data, tokenizer, max_len=200)
    test_params = {"batch_size": 16, "shuffle": False, "num_workers": 0}
    dataloader = DataLoader(dataset, **test_params)

    predictions = [model.forward(batch) for batch in dataloader]
    predictions = list(torch.cat(predictions).argmax(1).numpy())
    logging.info("predictions: {}".format(predictions))
    return predictions


# postprocess
def output_fn(predictions, content_type):
    assert content_type == "application/json"
    return json.dumps(predictions)
