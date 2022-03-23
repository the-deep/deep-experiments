import os, string

def clean_sentence(x):
    x = x.replace("\n", " ")
    x = x.translate(str.maketrans(' ', ' ', string.punctuation))
    return x

def prepare_data(df, column, filename=None):
    if not os.path.exists("./fast_data"):
        os.makedirs("./fast_data")
    total = []
    text = [c.strip().lower() for c in df.excerpt]
    target = [[a.strip().lower().replace(" ", "*") for a in c] if c else ["NEGATIVE"] for c in df[column].tolist()]
    for x, y in zip(text, target):
        x = clean_sentence(x)
        labels = " ".join([f"__label__{c}" for c in y])
        total.append(" ".join([labels, x]))
        
    a =  "\n".join(total)
    with open(f"./fast_data/{filename}", "w+") as f:
        f.write(a)