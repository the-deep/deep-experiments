Deep Experiments
================

This repository is dedicated to the NLP part of the DEEP project.
The code is tightly coupled with AWS Sagemaker.

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

Streamlit
~~~~~~~~~

We incorporated in the repo the ``streamlit`` web application. In the future we will put it in
another repo.

To use it locally:

.. code-block:: bash

    make streamlit-install
    streamlit run scripts/testing/subpillar_pred_with_st.py

You can also build and deploy a Docker application to ECR and Beanstalk:

.. code-block:: bash

    make streamlit-build
    make streamlit-deploy

You may need to change the local image name (WIP).
Also we plan to add Github Actions to automate this procedure


Folder structure
----------------

- ``data`` contains the data
- ``deep`` contains the code
- ``notebooks`` contains all the Jupyter Notebook, divided by category and person working on them
- ``scripts`` contains the training scripts necessary for Sagemaker
