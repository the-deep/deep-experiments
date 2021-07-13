Experiment tracking
====================

We use `MLFlow <https://www.mlflow.org>`_ to track experiments.
You can find `here <http://mlflow-deep-387470f3-1883319727.us-east-1.elb.amazonaws.com/#/>`_
all our experiments.

Training a model
----------------

In our configuration, MLFlow is tightly integrated with SageMaker.
If you have never used it before, we advise you to take the following step to take
confidence with our set-up.

Launch a training job
~~~~~~~~~~~~~~~~~~~~~~~~

Start from `this notebook <https://github.com/the-deep/deep-experiments/blob/main/notebooks/examples/pytorch-lightning-sectors.ipynb>`_.
Launching a job is not complicated, but many
