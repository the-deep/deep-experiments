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

streamlit-build:
	docker build . -t deatinor/streamlit --no-cache --platform=linux/amd64

streamlit-build-arm:
	docker build . -t deatinor/streamlit-m1 --no-cache

streamlit-deploy:
	aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 961104659532.dkr.ecr.us-east-1.amazonaws.com
	docker tag deatinor/streamlit 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit
	docker push 961104659532.dkr.ecr.us-east-1.amazonaws.com/streamlit
