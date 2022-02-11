from transformers import AutoTokenizer

def get_embeddings(
    entry: str,
    tokenizer=AutoTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased"), 
    max_length: int = 256
    ):
    """
    function to get tokenized sentence (without padding)
    """
    inputs = tokenizer(
        entry,
        None,
        truncation=True,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    final_ids = [ids[i] for i in range (max_length) if mask[i]]

    return final_ids