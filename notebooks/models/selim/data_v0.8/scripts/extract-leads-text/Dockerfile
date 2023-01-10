FROM python:3.9

# install project requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt && rm -f requirements.txt
RUN pip install git+https://github.com/the-deep/deepex

COPY pull_leads_batch.py pull_leads_batch.py

CMD ["python", "-u","pull_leads_batch.py"]  