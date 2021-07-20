from pathlib import Path
from cloudpathlib import CloudPath
import yaml

import boto3

current_path = Path(__file__).parent
config_path = current_path / "config.yml"

if __name__ == "__main__":
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = CloudPath(config["model"])

    s3 = boto3.client("s3")
    bucket_name = model.cloud_prefix + model.bucket
    s3.download_file(
        str(model.bucket), str(model).replace(bucket_name, ""), str(current_path / "conda.yaml")
    )
