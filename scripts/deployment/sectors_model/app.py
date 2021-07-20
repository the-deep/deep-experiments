import json

import mlflow
import pandas as pd

MODEL_PATH = "s3://deep-mlflow-artifact/2/f3b4e0f9a0364f8dbfe1563b248348a1/artifacts/model"
INPUT_PATH = "/opt/ml/processing/input.json"
OUTPUT_PATH = "/opt/ml/processing/output.json"


def load_model():
    return mlflow.pyfunc.load_model(MODEL_PATH)


def read_input() -> pd.DataFrame:
    with open(INPUT_PATH, "r") as f:
        input_data = json.load(f)

    input_data = pd.DataFrame({"excerpt": input_data})
    return input_data


def write_output(predictions: pd.Series):
    predictions = list(predictions)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    model = load_model()
    input_data = read_input()
    predictions = model.predict(input_data)
    write_output(predictions)
