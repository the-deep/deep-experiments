from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data"
IMMAP_PATH = DATA_PATH / "immap"
FRAMEWORKS_PATH = DATA_PATH / "frameworks_data"

SCRIPTS_PATH = ROOT_PATH / "scripts"
SCRIPTS_MODELS_PATH = SCRIPTS_PATH / "models"

SAGEMAKER_BUCKET = "deep-experiments-sagemaker-bucket"

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
