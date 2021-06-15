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

streamlit-install:
	pip install -r requirements/streamlit-requirements.txt
	pip install git+https://github.com/casics/nostril.git

streamlit-build:
	docker build . -f docker/Dockerfile -t streamlit --no-cache --platform=linux/amd64

streamlit-build-arm:
	docker build . -f docker/Dockerfile -t streamlit-m1 --no-cache

streamlit-deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
	docker tag streamlit 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit
	docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit

documentation:
	rm -rf docs/documentation
	sphinx-build -b html docs/source docs/documentation