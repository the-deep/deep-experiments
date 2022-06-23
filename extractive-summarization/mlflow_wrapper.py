import pickle

import spacy
import torch
import mlflow
import mlflow.pyfunc

from train import get_args, get_separate_layer_groups, LABEL_NAMES

tokenizer_path = "/output/tokenizer.pkl"


def serialize_entities(tokenizer):
    with open(tokenizer_path, "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# prepare the artifacts
state_dict_path = "/output/state_dict_best_model.pt"
artifacts = {
    "state_dict_model": state_dict_path,
    "tokenizer": tokenizer_path,
}

# prepare the conda environment
conda_env = {
    "channels": ["defaults"],
    "dependencies": [
        "python=3.9.0",
        {
            "pip": [
                f"mlflow=={mlflow.__version__}",
                f"torch=={torch.__version__}",
                f"spacy=={spacy.__version__}",
            ]
        },
    ],
    "name": "extractive-summarization",
}


class ModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from model import Model

        args, _ = get_args()
        # load in and deserialize the model tokenizer
        with open(context.artifacts["tokenizer"], "rb") as handle:
            self._model_tokenizer = pickle.load(handle)

        self._model = Model(
            args.model_name_or_path,
            self._model_tokenizer,
            num_labels=len(LABEL_NAMES),
            token_loss_weight=args.token_loss_weight,
            loss_weights=args.loss_weights,
            slice_length=args.max_length,
            extra_context_length=args.extra_context_length,
            n_separate_layers=args.n_separate_layers,
            separate_layer_groups=get_separate_layer_groups(args),
        )
        self._model.load_state_dict(
            torch.load(context.artifacts["state_dict_model"], map_location="cpu")
        )
        self._model.eval()

    def predict(self, context, input_model):
        print(input_model)
        sentence = input_model
        sentence_tokens = self._model_tokenizer.texts_to_sequences([sentence])

        inputs = torch.tensor(sentence_tokens)
        inputs = inputs.type(torch.LongTensor)
        inputs = inputs.to("cpu")
        model_pred = self._model(inputs).detach().cpu().numpy()
        predictions = self._model(model_pred)
        pred_results = predictions[0].tolist()
        results = {
            "toxicity_score": [pred_results[0]],
            "target": [pred_results[1]],
            "severe_toxicity": [pred_results[2]],
            "obscene": [pred_results[3]],
            "identity_attack": [pred_results[4]],
            "insult": [pred_results[5]],
            "threat": [pred_results[6]],
        }
        return results


model_path = "/outputs/"


# Package the model!
mlflow.pyfunc.save_model(
    path=model_path,
    python_model=ModelWrapper(),
    artifacts=artifacts,
    conda_env=conda_env,
    code_path=[
        "/model.py",
        "/new_training.py",
        "/mlflow_wrapper.py",
    ],
)
