import os
import urllib
import zipfile
from typing import List
import numpy as np
import onnxruntime.backend as backend
from onnx import load


class Derand:
    # This is a non-optimized Python port of Derand
    # https://github.com/Netflix/derand
    MODEL_CHARS = [
        "!",
        '"',
        "#",
        "$",
        "%",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        ":",
        ";",
        "<",
        "=",
        ">",
        "?",
        "@",
        "_",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    CHAR_TO_INDEX = {char: i for i, char in enumerate(MODEL_CHARS)}
    MAX_LEN = 16
    RND = "<rnd>"
    MODEL_URL = "https://randomly-public-us-east-1.s3.amazonaws.com/derand.onnx.zip"
    MODEL_ZIP = "derand.onnx.zip"
    MODEL_PATH = "derand.onnx"

    def __init__(self):
        self.load_model()

    def download_model(self):
        if not os.path.exists(Derand.MODEL_PATH):
            urllib.request.urlretrieve(Derand.MODEL_URL, Derand.MODEL_ZIP)
            with zipfile.ZipFile(Derand.MODEL_ZIP, "r") as zip_ref:
                zip_ref.extract(Derand.MODEL_PATH)

    def load_model(self):
        self.download_model()
        self.model = backend.prepare(load(self.MODEL_PATH), "CPU")

    def pred(self, input):
        return self.model.run(input)

    def process_input(self, input: str) -> np.ndarray:
        if input is None or not len(input):
            return np.zeros((1, self.MAX_LEN), dtype=np.float32)
        if len(input) > self.MAX_LEN:
            input = input[-self.MAX_LEN :]
        elif len(input) < self.MAX_LEN:
            input = (self.MAX_LEN - len(input)) * self.MODEL_CHARS[0] + input
        input = input.lower()
        result = [self.CHAR_TO_INDEX.get(char, 0) for char in input]
        return np.array(result, dtype=np.float32).reshape(1, -1)

    def process_output(self, output):
        return output[0].argmax() == 1

    def is_empty(self, text: str):
        return (text is None) or (text == "") or (text.strip() == "")

    def predict_randomness_per_word(self, words: List[str]) -> List[bool]:
        return [self.process_output(self.pred(self.process_input(word))) for word in words]

    def classify(self, text: str):
        if self.is_empty(text):
            return []
        return self.predict_randomness_per_word(text.split(" "))

    def clean(self, text: str) -> str:
        if self.is_empty(text):
            return ""

        result = []
        for tokenized_word in self.tokenize_words(text.split(" ")):
            if tokenized_word != self.RND:
                result.append(tokenized_word)

        return " ".join(result)

    def tokenize(self, text: str) -> str:
        if self.is_empty(text):
            return ""

        result = []
        for tokenized_word in self.tokenize_words(text.split(" ")):
            result.append(tokenized_word)

        return " ".join(result)

    def tokenize_words(self, words: List[str]) -> List[str]:
        random_mask = self.predict_randomness_per_word(words)
        return [self.RND if is_rnd else word for word, is_rnd in zip(words, random_mask)]


if __name__ == "__main__":
    derander = Derand()
    print(derander.tokenize("hello 3y29842ysjhfs world"))
    # "hello <rnd> world"

    print(derander.clean("hello 3y29842ysjhfs world"))
    # "hello world

    print(derander.clean("Hello 3y29842ysjhfs World!"))
