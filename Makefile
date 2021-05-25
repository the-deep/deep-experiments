install:
	pip install -r requirements.txt
	pre-commit install

local-install: install
	pip install -r local-requirements.txt