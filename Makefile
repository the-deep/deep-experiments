cloud-install:
	pip install -r requirements/cloud-requirements.txt
	pre-commit install

install: cloud-install
	pip install -r requirements/local-requirements.txt
	conda install -y jupyter
	conda install -y -c conda-forge jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user

doc-install:
	pip install -r requirements/doc-requirements.txt

dev-install: install doc-install

documentation:
	rm -rf docs/documentation
	sphinx-build -b html docs/source docs/documentation