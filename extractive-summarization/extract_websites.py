from deep_parser import TextFromWeb
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from dataclasses import dataclass
from transformers.hf_argparser import HfArgumentParser

@dataclass
class Args:
    leads_csv_path: str
    output_path: str

def main(args):
    out_dir = Path(args.output_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.leads_csv_path)

    for (project_id, lead_id), group in tqdm(list(df.groupby(["project_id", "lead_id"]))):
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
    
    main(args)