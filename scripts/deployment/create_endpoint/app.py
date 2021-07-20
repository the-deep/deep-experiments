import mlflow.sagemaker

logged_model = "s3://deep-mlflow-artifact/2/f3b4e0f9a0364f8dbfe1563b248348a1/artifacts/model"
SAGEMAKER_ROLE_ARN = (
    "arn:aws:iam::961104659532:role/service-role/AmazonSageMaker-ExecutionRole-20210519T102514"
)

if __name__ == "__main__":
    mlflow.sagemaker.deploy(
        "pl-example",
        logged_model,
        execution_role_arn=SAGEMAKER_ROLE_ARN,
        image_url="961104659532.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:latest",
        region_name="us-east-1",
        instance_type="ml.p2.xlarge",
        synchronous=False,
        archive=True,
    )
