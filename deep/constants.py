from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data"
IMMAP_PATH = DATA_PATH / "immap"

SCRIPTS_PATH = ROOT_PATH / "scripts"
SCRIPTS_MODELS_PATH = SCRIPTS_PATH / "models"

SAGEMAKER_BUCKET = "deep-experiments-sagemaker-bucket"
