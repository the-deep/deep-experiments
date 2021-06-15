Terraform
==========

We currently use `Terraform <https://www.terraform.io>`_ to configure the infrastructure.

What is on terraform?
---------------------

- All S3 buckets, except the Terraform state
- The Sagemaker IAM. We do not store on Terraform the rest of the IAM
- The Sagemaker Notebook Instances and Github repository
- The Streamlit ECR repository

What is not on Terraform?
-------------------------

- The IAM, except the Sagemaker one
- The Beanstalk environments
- The Terraform state