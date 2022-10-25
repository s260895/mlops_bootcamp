FROM python:3.10

RUN apt update -y && apt upgrade -y
RUN pip install -U pip
RUN git clone https://github.com/s260895/mlops_bootcamp.git
RUN git init
WORKDIR /mlops_bootcamp
RUN pip install -r requirements.txt 
RUN pipenv shell
RUN mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=./mlruns
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "04-5_flask_app:app" ]