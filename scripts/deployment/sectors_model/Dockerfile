FROM python:3.9.5

COPY scripts/deployment/sectors_model app/
WORKDIR app/
#RUN apt-get update && apt-get install make && apt-get install gcc && apt-get install g++
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
