Deep Experiments
================

This repository is dedicated to the NLP part of the DEEP project.
The code is tightly coupled with AWS Sagemaker.

You can access the full documentation
`here <http://deep-documentation.s3-website-us-east-1.amazonaws.com/deep-experiments/index.html>`_
(if you are already browsing the documentation on the web, it links to the same page you are at now)

Quick-start
-----------

Local development
~~~~~~~~~~~~~~~~~

Contact Stefano to get the AWS credentials, install the
`AWS CLI <https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html>`_

Clone the repo and pull the data

.. code-block:: bash

    git clone <deep_experiments_repo>
    cd deep-experiments

Create a new conda environment:

.. code-block:: bash

   conda create -n deepl python=3.9.1

Install necessary libraries:

.. code-block:: bash

    make dev-install

Pull the data:

.. code-block:: bash

    dvc pull




Notebook instances on AWS
~~~~~~~~~~~~~~~~~~~~~~~~~

Ask Stefano for a AWS user account and a new Notebook instance on AWS.
The notebook instance will be created in the region ``us-east-1``, switch to this
region if you don't find your instance.
The notebook instance comes with the repo already cloned.

Once it is ready, start the instance and click on *Open Jupyter*.
Open the Jupyter terminal and ``cd`` to the ``deep-experiments`` repo. It should be:

.. code-block:: bash

    cd SageMaker/deep-experiments

Run:

.. code-block:: bash

    make cloud-install

(This must be run everytime the instance is activated)

Pull the data:

.. code-block:: bash

    dvc pull

Folder Structure
----------------

- ``data`` contains the data
- ``deep`` contains the code
- ``docker`` contains the Dockerfile used to build the test environment.
- ``notebooks`` contains all the Jupyter Notebook, divided by category and person working on them
- ``scripts`` contains the training scripts necessary for Sagemaker
- ``requirements`` contains all the Python requirements for the different configurations