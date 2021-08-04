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

- Start from `this notebook <https://github.com/the-deep/deep-experiments/blob/main/notebooks/examples/pytorch-lightning-sectors.ipynb>`_ to launch a training job
- The training job launches a script that you can find in `this folder <https://github.com/the-deep/deep-experiments/tree/main/scripts/examples/sector-pl>`_.
- Start looking at the file ``train.py``. Familiarize with the script, try to launch your own.
- Check the training results on SageMaker -> Training -> Training jobs.
- If the training failed, check the training logs to understand what went wrong

Overall, it's important that you are sufficiently confident that the script is running without errors.
Running a training takes at least 8 minutes to start the machine. Debugging a script can
become easily a long process.

Deploying a model
-----------------

One of the advantages of using MLFlow is that the deployment is made easy by its integration
with SageMaker.
You can find an example of deployment in the repo.

- As for the training start from the `deployment notebook <https://github.com/the-deep/deep-experiments/blob/main/notebooks/examples/pytorch-lightning-inference.ipynb>`_
- The deployment code is in the same folder as the `training one <https://github.com/the-deep/deep-experiments/tree/main/scripts/examples/sector-pl>`_
- The key of the deployment is creating a class that inherits from `mlflow.pyfunc.PythonModel` with a `predict()` function.
- That class is pickled and logged as artifact of the training. At inference time it will be used to make predictions.

Additionally, consider the following for more configurable deployment:

- *Dynamic inference parameters*: Store inference hyperparameters (e.g., batch size or thresholds) as a separate artifact in MLFlow.  Use `artifacts` options in `log_model` and then retrieve the file using the `context` object provided by the MLFlow in `load_context` or `predict`.
- *Multiple outputs*: `predict` function can return a Pandas DataFrame object. Employ it if the model has multiple targets or for providing logits scores for dynamic threshold adjusting on the client-side.
- *Serving labels*: Log a separate artifact in MLFlow for the client-side to map predictions back to human-readable labels.
