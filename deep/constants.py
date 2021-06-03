from pathlib import Path
from cloudpathlib import CloudPath

ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data"
IMMAP_PATH = DATA_PATH / "immap"
FRAMEWORKS_PATH = DATA_PATH / "frameworks_data"
LATEST_DATA_PATH = FRAMEWORKS_PATH / "data_v0.4.3"

SCRIPTS_PATH = ROOT_PATH / "scripts"
SCRIPTS_TRAINING_PATH = SCRIPTS_PATH / "training"
SCRIPTS_INFERENCE_PATH = SCRIPTS_PATH / "inference"

DEV_BUCKET = CloudPath("s3://sagemaker-deep-experiments-dev")
PROD_BUCKET = CloudPath("s3://sagemaker-deep-experiments-prod")

DIMENSION_CLASSES = [
    "Shock Informaton",
    "Effects Systems And Networks",
    "Effects On Population",
    "Capacities & Response",
    "At Risk",
    "Scope & Scale",
    "Impact",
    "Humanitarian Conditions",
    "Priorities",
    "Context",
]

SECTOR_CLASSES = [
    "Agricolture",
    "Cross",
    "Education",
    "Food Security",
    "Health",
    "Livelihoods",
    "Logistics",
    "Nutrition",
    "Protection",
    "Shelter",
    "Wash",
]
