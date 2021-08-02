FROM continuumio/miniconda3

COPY scripts/deployment/sectors_model app/
WORKDIR app/
RUN conda init bash
RUN conda env create --name model -f conda.yaml
RUN which python
#RUN apt-get update && apt-get install make && apt-get install gcc && apt-get install g++
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "model", "python", "app.py"]