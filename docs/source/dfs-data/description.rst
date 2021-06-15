Data
====

The data for the project is in the folder `data`.
The `immap` subfolder contains only IMMAP data, while `frameworks_data` contains all the rest.

What is our data?
------------------------

Our data is composed of ~100.000 sentences extracted from PDF and web articles. Each sentence has been manually
labeled by taggers. Multiple labels are associated to the same sentence:

- Sector
- Pillar
- Subpillar

We want to predict all of them. Each subpillar belongs to one and only one pillar.
More labels are coming.

How good is the data
--------------------

Not so much, some classes are ambiguous and, since we have multiple taggers, their tags are not
always consistent.
However, we have a lot of data, which is good.


Which data should I use?
------------------------

We are currently working only with `frameworks_data` and the most recent data version.

We advise you to import the file:

.. code-block:: python

    from deep.constants import *

The variable ``LATEST_DATA_PATH`` points to the most recent version of the data.

What are the differences between the data versions?
----------------------------------------------------

Alongside the dataset queried (IMMAP vs all) mainly bug fixes and better definition of the classes.
Please use the latest version.

How do I get the most recent version?
-------------------------------------

We use `DVC <https://dvc.org>`_ to deal with data. Simply run

.. code-block:: bash

    dvc pull

to get the most recent version.

