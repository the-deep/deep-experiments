Deployment
===========

To deploy a trained model we use the following tools:

- SageMaker training jobs
- MLFlow
- ECR Docker Image
- AWS Lambda functions, managed via Terraform
- SageMaker processing job

In the following we describe one by one the interaction between these parts.

SageMaker training jobs and MLFlow
-----------------------------------

Models are trained in SageMaker training jobs. To use a trained model in production we need to
store it somewhere after training.
We use the MLFlow built-in function ``log_model``. You can see a usage example
`here <https://github.com/the-deep/deep-experiments/blob/main/scripts/examples/sector-pl/train.py#L102>`_.

There are a couple of consideration to do when logging your model:

- You cannot simply log your pytorch model
- You need to store also dependencies alongside the model.

Storing the appropriate model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A pytorch model can be automatically logged in MLFlow via ``mlflow.pytorch.log_model()``.
Logging a model with this syntax makes it extremely easy to use it in production, with the following syntax:

.. code-block:: python

    loaded_model = mlflow.pytorch.load_model(<model_s3_url>)
    loaded_model.predict(<data>)

The problem is the data: it supports only data in a tensor format, that in the NLP world means already tokenized.
We need to embed the tokenization into the model.

To do so, we subclass the class ``mlflow.pyfunc.PythonModel``, we create a suitable ``predict()`` function with
the inference logic and we log that model. In the previous example the logged class is
`this one <https://github.com/the-deep/deep-experiments/blob/040617759e20fd629bc669939999fcacfbaa19db/scripts/examples/sector-pl/inference.py#L13>`_
and it accepts as input a list of strings in json format.

Once you have logged that class, to use it at prediction time, it is sufficient to run:

.. code-block:: python

    loaded_model = mlflow.pyfunc.load_model(<model_s3_url>)
    loaded_model.predict(<data>)

This time the data can be whatever you want, in this case a list of strings.

Storing dependencies
~~~~~~~~~~~~~~~~~~~~

Dependencies are both packages dependencies and code files dependencies.
Refer to `here <https://github.com/the-deep/deep-experiments/blob/040617759e20fd629bc669939999fcacfbaa19db/scripts/examples/sector-pl/train.py#L105>`_
to see how dependencies are added to the model artifact.

You can reuse the same code, modifying of course ``code_path``.

ECR Docker Image
----------------

We have two docker images, for live and batch predictions.
They are generated via the Github CI in the repo ``deep-deployment``. The two docker images,
named ``batch-models`` and ``live-models`` should work for any type of model.
Start working with the live one, the batch one is still a WIP.

Live models
~~~~~~~~~~~

The live models docker image is only a tiny wrapper around the function ``mlflow.sagemaker.deploy``
that deploys a SageMaker endpoint starting from a logged model.

This should work for **any model**. It works via 2 environment variables that should be passed to
the docker images when it is run:

- ENDPOINT_NAME, the name of the endpoint
- MODEL_PATH, the s3 path of the model

AWS Lambda and SageMaker Processing job
----------------------------------------

Live models
~~~~~~~~~~~

We use AWS Lambda to create, delete and inspect an endpoint.
At the moment we have 3 functions:

- ``create-endpoint``
- ``delete-endpoint``
- ``endpoint-status``

You need to create your own one, using Terraform, starting from the same templates in the folder ``deep``.
The only thing you should change are the environment variables that are passed to it.

The aim of the ``create-endpoint`` is to launch the live models docker image discussed before.
Unfortunately, it is not possible to call ``mlflow.sagemaker.deploy`` inside an AWS Lambda for the requirements
that are too low. We are thus forced to use SageMaker processing jobs to launch it.
