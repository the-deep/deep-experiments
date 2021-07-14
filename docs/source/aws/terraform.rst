Terraform
==========

We currently use `Terraform <https://www.terraform.io>`_ to configure the infrastructure.
`This <https://github.com/the-deep/dfs-deepl-terraform>`_ is the repository.

What is on terraform?
---------------------

- All S3 buckets, except the Terraform state
- The Sagemaker IAM. We do not store on Terraform the rest of the IAM
- The Sagemaker Notebook Instances and Github repository
- The test environments ECR repository
- The MLFlow server and ECR repository

What is not on Terraform?
-------------------------

- The IAM, except the Sagemaker one
- The Beanstalk environments
- The Terraform state