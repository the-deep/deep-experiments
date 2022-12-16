from deep_parser import TextFromFile, TextFromWeb
from deep_parser.helpers.errors import DocumentProcessingError
from tqdm.auto import tqdm
from glob import glob
import base64
from pathlib import Path
import timeout_decorator
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Args:
    leads_csv_path: str
    input_path: str
    output_path: str


@timeout_decorator.timeout(5 * 60, use_signals=False)
def extract(in_path, out_path):
    if out_path.exists():
        return False

    with open(in_path, "rb") as f:
        binary = base64.b64encode(f.read())

    try:
        document = TextFromFile(stream=binary, ext="pdf")
        text, other = document.extract_text()
    except (RuntimeError, DocumentProcessingError):
        return False

    try:
        open(out_path, "w").write(text)
    except UnicodeEncodeError:
        return False

    return True


def work(in_path, out_path):
    try:
        return extract(in_path, out_path)
    except timeout_decorator.TimeoutError:
        print(f"Timeout for File {in_path}!")


def pull_pdfs(args, leads_df: pd.DataFrame):
    print("###################### Start puling pdf data.")
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    name_to_id = {
        row["url"].rstrip("/").split("/")[-1]: (int(row["id"]), int(row["project_id"]))
        for _, row in leads_df.iterrows()
        if row["url"] is not np.nan
    }

    paths = [Path(p) for p in glob(str(input_path / "*.pdf"))]

    for in_path in tqdm(paths):
        name = Path(in_path).name
        lead_id, project_id = name_to_id[name]

        out_path = output_path / str(project_id) / f"{lead_id}.txt"
        out_path.parent.mkdir(exist_ok=True, parents=True)

        work(in_path, out_path)


def pull_websites(args, leads_df: pd.DataFrame):
    print("###################### Start puling websites data.")
    out_dir = Path(args.output_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    for (project_id, lead_id), group in tqdm(
        list(leads_df.groupby(["project_id", "lead_id"]))
    ):
        project_id, lead_id = str(int(project_id)), str(int(lead_id))
        urls = group["url"].unique()

        assert len(urls) == 1

        url = urls[0]

        if not isinstance(url, str):  # possibly np.nan
            continue

        if url.endswith(".pdf"):
            print(f"{url} is a PDF! Skipping...")
            continue

        nested_dir_path = out_dir / project_id
        nested_dir_path.mkdir(exist_ok=True, parents=True)

        file_path = nested_dir_path / f"{lead_id}.txt"
        if file_path.exists():
            continue

        text = None

        try:
            parser = TextFromWeb(url=url)
            text = parser.extract_text()
            parser.close()
        except:
            continue

        open(file_path, "w").write(text)


if __name__ == "__main__":
    (args,) = HfArgumentParser([Args]).parse_args_into_dataclasses()
    leads_df = pd.read_csv(args.excerpts_data_path)[
        ["project_id", "lead_id", "url"]
    ].drop_duplicates()

    pull_websites(args, leads_df)
    pull_pdfs(args, leads_df)
