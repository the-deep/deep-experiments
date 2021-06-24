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

offline-test-env-install:
	pip install -r requirements/test-env-requirements.txt
	pip install git+https://github.com/casics/nostril.git

offline-test-env-build:
	docker build . -f docker/Dockerfile.offline_test_env -t offline_test_env --no-cache --platform=linux/amd64

offline-test-env-build-arm:
	docker build . -f docker/Dockerfile.offline_test_env -t offline_test_env-m1 --no-cache

offline-test-env-deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
	docker tag offline_test_env 961104659532.dkr.ecr.us-east-1.amazonaws.com/offline_test_env
	docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/offline_test_env

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