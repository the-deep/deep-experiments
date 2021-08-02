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

mlflow-build:
	docker build . -f docker/Dockerfile.mlflow -t mlflow --no-cache --platform=linux/amd64

mlflow-build-arm:
	docker build . -f docker/Dockerfile.mlflow -t mlflow --no-cache

mlflow-deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
	docker tag mlflow 961104659532.dkr.ecr.us-east-1.amazonaws.com/mlflow
	docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/mlflow

documentation:
	rm -rf docs/documentation
	sphinx-build -b html docs/source docs/documentation