cloud-install:
	pip install -r cloud-requirements.txt
	pre-commit install

install: cloud-install
	pip install -r requirements.txt
	conda install -y jupyter
	conda install -y -c conda-forge jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user

dev-install: install
	pip install -r doc-requirements.txt

streamlit-install:
	pip install -r streamlit-requirements.txt
	pip install git+https://github.com/casics/nostril.git

streamlit-build:
	docker build . -t deatinor/streamlit --no-cache --platform=linux/amd64

streamlit-build-arm:
	docker build . -t deatinor/streamlit-m1 --no-cache

streamlit-deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
	docker tag deatinor/streamlit 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit
	docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit

documentation:
	sphinx-build -b html docs/source docs/documentation

documentation-push:
	aws s3 sync docs/documentation s3://deep-documentation/deep-experiments