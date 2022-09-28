import torch


def prepare_data_for_forward_pass(
    batch,
    slice_length: int,
    extra_context_length: int,
    pad_token_id: int,
    num_labels: int,
    training: bool,
):

    """
    batch: same structure as in the '_operate_train_or_val_step' function.
    training: bool: whether we are training (the are present labels) or not (no loss computation needed)
    """

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    token_labels_mask = batch["token_labels_mask"]
    token_labels = batch["token_labels"]
    length = input_ids.shape[0]

    n_steps = int(length / slice_length)

    extra_context = torch.cat(
        [
            torch.full(
                extra_context_length,
                pad_token_id,
                device=input_ids.device,
            ),
            input_ids[: length - extra_context_length],
        ],
        1,
    ).view(n_steps, slice_length)[:, :extra_context_length]

    input_ids = input_ids.view(n_steps, slice_length)
    attention_mask = attention_mask.view(n_steps, slice_length)

    # Adding extra context
    input_ids = torch.cat([extra_context, input_ids], 1)
    attention_mask = torch.cat([torch.ones_like(extra_context), attention_mask], 1)

    if training:
        token_labels_mask = torch.cat(
            [
                torch.zeros_like(extra_context),
                token_labels_mask.view(n_steps, slice_length),
            ],
            1,
        )
        token_labels = torch.cat(
            [
                torch.zeros((*extra_context.shape, num_labels))
                .type_as(token_labels)
                .to(extra_context.device),
                token_labels.view((n_steps, slice_length, num_labels)),
            ],
            1,
        )
    else:
        token_labels_mask = None
        token_labels = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_labels_mask": token_labels_mask,
        "token_labels": token_labels,
    }
