Sagemaker
=========

We use mainly AWS Sagemaker as training and deployment platform.
Many features have been built into the platform. We go over the ones we use:

Notebook Instances
-------------------

The easiest one, mainly EC2 + jupyter notebook already installed. It should not be difficult to user

They have already integrated Git repositories.

Training Jobs
-------------

A useful feature to train models quickly and cheaply.
An example notebook is ``ssa-1.8-fastai-multilabel-remote.ipynb`` with the corresponding scripts in
``scripts/training/stefano/multiclass-fastai``.

There are many details in the notebook and in the scripts that I will write here.

Hyperparameter tuning jobs
--------------------------

Very similar to training jobs, they allow to tune the hyperparamers, should not be difficult to use,
but we did not try yet.

Inference - API
---------------

Deploy quickly one of the models trained in the trained job.

Inference - Batch Transform Job
-------------------------------
Associate to one of the models trained in the trained job a corresponding inference job,
that can be triggered. This is probably what we will use in the future for production.

