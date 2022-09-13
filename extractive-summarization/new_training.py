from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import time
from copy import copy

import mlflow

from rouge import Rouge
from transformers import AutoTokenizer

from train import get_test_train_data, get_args, LABEL_NAMES, get_separate_layer_groups
from model import Model


class ExtractionDataset(Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __getitem__(self, idx):
        d = self.dset[idx]
        return {
            "id": d["id"],
            "input_ids": d["input_ids"],
            "attention_mask": d["attention_mask"],
            "offset_mapping": d["offset_mapping"],
            "token_labels": d["token_labels"],
            "token_labels_mask": d["token_labels_mask"],
            "sentence_labels": d["sentence_labels"],
            "sentence_labels_mask": d["sentence_labels_mask"],
        }

    def __len__(self):
        return len(self.dset)


def train_model(
    model, train_dataset, test, loss_fn, output_dim, lr=0.001, batch_size=512, n_epochs=10
):
    param_lrs = [{"params": param, "lr": lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6**epoch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    training_loss = []
    validation_loss = []
    avg_val_loss = 0
    best_loss = float("inf")
    # checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    best_state_dict = None
    for epoch in range(n_epochs):
        start_time = time.time()

        scheduler.step()
        model.train()
        avg_loss = 0

        for data in tqdm(train_loader, disable=False):
            x_batch = data[:-1]
            y_batch = data[-1]
            y_pred = model(*x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        training_loss.append(avg_loss)
        model.eval()
        # test_preds = np.zeros((len(test), output_dim))

        avg_val_loss = 0
        for _, val_data in enumerate(test_loader):
            x_batch = val_data[:-1]
            y_batch = val_data[-1]
            y_pred = model(*x_batch)
            val_loss = loss_fn(y_pred, y_batch)
            avg_val_loss += val_loss.item() / len(test_loader)
            # y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())

        elapsed_time = time.time() - start_time
        validation_loss.append(avg_val_loss)
        if avg_val_loss < best_loss:
            print("saving the best model so far")
            state_dict_path = "/output/state_dict_best_model.pt"
            torch.save(model.state_dict(), state_dict_path)
            best_state_dict = copy(model.state_dict())
            best_loss = avg_val_loss
        print(
            f"Epoch {epoch + 1}/{n_epochs}\t training_loss={avg_loss:.4f}"
            f"\t validation_loss={avg_val_loss: 4f} \t "
            f"time={elapsed_time:.2f}s"
        )
    mlflow.log_param("best_loss", best_loss)
    mlflow.log_param("last_epoch_loss", avg_val_loss)
    mlflow.pytorch.log_model(best_state_dict, "extraction-model")

    return training_loss, validation_loss


def main():
    args, training_args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = Model(
        args.model_name_or_path,
        tokenizer,
        num_labels=len(LABEL_NAMES),
        token_loss_weight=args.token_loss_weight,
        loss_weights=args.loss_weights,
        slice_length=args.max_length,
        extra_context_length=args.extra_context_length,
        n_separate_layers=args.n_separate_layers,
        separate_layer_groups=get_separate_layer_groups(args),
    )
    loss_fn = Rouge(metrics=["rouge-1", "rouge-2", "rouge-l"]).get_scores
    output_dim = None
    train_data, test_data = get_test_train_data(args, training_args)
    train_dataset, test_dataset = ExtractionDataset(train_data), ExtractionDataset(test_data)
    train_model(
        model, train_dataset, test_dataset, loss_fn, output_dim, lr=0.001, batch_size=5, n_epochs=10
    )


def _debug_arrow_dataset(dataset):
    for i, x in enumerate(dataset):
        print(f"Data entry {i+1}")
        for k, v in x.items():
            if isinstance(v, list):
                print(k, (len(v),))
            elif isinstance(v, str):
                print(k, (len(v),), v)
            else:
                try:
                    print(k, v.shape)
                except ValueError:
                    print(k, v)
        print()


if __name__ == "__main__":
    main()
