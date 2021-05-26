install:
	pip install -r requirements.txt
	pre-commit install

local-install: install
	pip install -r local-requirements.txt
	conda install -y jupyter
	conda install -y -c conda-forge jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user
