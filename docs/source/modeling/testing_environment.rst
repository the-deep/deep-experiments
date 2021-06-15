Testing Environment
===================

We have built a simple online test environment to check the predictions of our models on the subpillars.
You can find it `here <http://test-env.eba-crsiq2wb.us-east-1.elasticbeanstalk.com>`_.

It is useful to show to externals our results. We plan to add more features to it.

We have used Streamlit as Python library, and Docker + ECR + Beanstalk as deployment option.

Deployment
-----------------------

We use the Github CI to build the test environment image and push it to ECR.
We then use Beanstalk to serve it.

Building the image
~~~~~~~~~~~~~~~~~~

The image is automatically built via the CI.
If you want to change the code, check ``docker/Dockerfile`` and change the variables
``ENTRYPOINT`` and ``CMD``.
Currently the CI downloads the data from
``s3://<DEV_BUCKET>/test-environment/data`` to ``data/streamlit``. If you want to change
the input data, look at that.

Pushing to ECR
~~~~~~~~~~~~~~

It is fully automated and a priori you should not do anything

Update the image on Beanstalk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is not automated, and you should do it **every time you want to serve a new image**.
Go in Beanstalk -> Environments -> Test env -> Upload and deploy.
Upload the file ``docker/Dockerrun.aws.json``.
