from pathlib import Path
from cloudpathlib import CloudPath

MLFLOW_SERVER = "http://mlflow-deep-387470f3-1883319727.us-east-1.elb.amazonaws.com/"
SAGEMAKER_ROLE = "AmazonSageMaker-ExecutionRole-20210519T102514"
SAGEMAKER_ROLE_ARN = (
    "arn:aws:iam::961104659532:role/service-role/" "AmazonSageMaker-ExecutionRole-20210519T102514"
)

ROOT_PATH = Path(__file__).parent.parent.absolute()
DATA_PATH = ROOT_PATH / "data"
IMMAP_PATH = DATA_PATH / "immap"
FRAMEWORKS_PATH = DATA_PATH / "frameworks_data"
LATEST_DATA_PATH = FRAMEWORKS_PATH / "data_v0.5"

SCRIPTS_PATH = ROOT_PATH / "scripts"
SCRIPTS_EXAMPLES_PATH = SCRIPTS_PATH / "examples"
SCRIPTS_TRAINING_PATH = SCRIPTS_PATH / "training"
SCRIPTS_INFERENCE_PATH = SCRIPTS_PATH / "inference"

DEV_BUCKET = CloudPath("s3://sagemaker-deep-experiments-dev")
PROD_BUCKET = CloudPath("s3://sagemaker-deep-experiments-prod")

SECTORS = [
    "Agriculture",
    "Cross",
    "Education",
    "Food Security",
    "Health",
    "Livelihoods",
    "Logistics",
    "Nutrition",
    "Protection",
    "Shelter",
    "WASH",
]

PILLARS = [
    "Humanitarian Conditions",
    "Capacities & Response",
    "Impact",
    "Priority Interventions",
    "People At Risk",
    "Priority Needs",
]

SUBPILLARS = [
    "Capacities & Response->International Response",
    "Capacities & Response->National Response",
    "Capacities & Response->Number Of People Reached",
    "Capacities & Response->Response Gaps",
    "Humanitarian Conditions->Coping Mechanisms",
    "Humanitarian Conditions->Living Standards",
    "Humanitarian Conditions->Number Of People In Need",
    "Humanitarian Conditions->Physical And Mental Well Being",
    "Impact->Driver/Aggravating Factors",
    "Impact->Impact On People",
    "Impact->Impact On People Or Impact On Services",
    "Impact->Impact On Services",
    "Impact->Impact On Systems And Services",
    "Impact->Number Of People Affected",
    "People At Risk->Number Of People At Risk",
    "People At Risk->Risk And Vulnerabilities",
    "Priority Interventions->Expressed By Humanitarian Staff",
    "Priority Interventions->Expressed By Population",
    "Priority Needs->Expressed By Humanitarian Staff",
    "Priority Needs->Expressed By Population",
]
