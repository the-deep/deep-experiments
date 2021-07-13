Current Work
============

Results
--------

An (incomplete) summary of the results of the models that we have tried can be found
`here <https://docs.google.com/spreadsheets/d/1zCbyZNb-Smz3GsEeJO6oyodvEK3rjgxJSDiC47jSh-o/edit#gid=299270945>`_

We tested the following:

- sectors
- subpillars

Sectors
~~~~~~~~~

The current performance is already good enough to test it in production.

Subpillars
~~~~~~~~~~

The current performance is not too good. It is ok for the pillars but not for the subpillars, in particular
the less frequent ones.

Models
~~~~~~

We tested recent Deep Learning models, in particular transformers, finetuned for text classification.
We tried basic techniques for unbalanced text classification (oversampling, entailment) with no success.
The best results so far have been obtained with a multi-lingual transformer.

Metric
~~~~~~

We give results of F1 score, recall and precision.
Recall may be more important, because the biggest goal would be to avoid the taggers to open and read the
PDF/web article. Only a good recall can give this.

Training
---------

You have two options to train a model:

- Use a Sagemaker notebook instance with GPU
- Use the `Sagemaker training feature <https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-training.html>`_ (recommended)

The first way is more intuitive and interactive.
However, once you set-up the main parameters of the training
and you reach a semi-definitive version of the training script, we advise you to move to training jobs.
They can use faster GPUs, are cheaper (you can run a training job from your
local PC and pay only the training time) and allow for a better tracking of the results and model artifact.

