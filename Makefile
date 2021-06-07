install:
	pip install -r requirements.txt
	pre-commit install

local-install: install
	pip install -r local-requirements.txt
	conda install -y jupyter
	conda install -y -c conda-forge jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user

cloud-install:
	source activate pytorch_p36
	local-install

streamlit:
	pip install -r streamlit-requirements.txt
	pip install git+https://github.com/casics/nostril.git
