FROM continuumio/anaconda3

COPY scripts/deployment/sectors_model app/
WORKDIR app/
#RUN apt-get update && apt-get install make && apt-get install gcc && apt-get install g++
RUN conda env create --name model --file=conda.yaml
RUN source activate model
ENTRYPOINT ["python"]
CMD ["app.py"]
