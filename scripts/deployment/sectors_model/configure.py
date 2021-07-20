from pathlib import Path
import yaml

import boto3

current_path = Path(__file__).parent
config_path = current_path / "config.yml"

if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = config["model"]
    dependencies = model + "/conda.yaml"

    s3 = boto3.client("s3")
    s3.download_file(model, "conda.yaml", str(current_path / "conda.yaml"))
