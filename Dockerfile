FROM python:3.10

RUN apt update && apt upgrade

RUN useradd -u 5000 mlops

USER mlops:mlops
WORKDIR /app
COPY ["requirements.txt", "./"]
COPY ["./data","./"]
COPY ["./mlruns","mlflow.db","./"]
COPY ["04-1_clean.py","04-2_train.py","04-3_predict.py","04_5-flask_app.py","./"]


RUN pip install -U pip
RUN pip install pipenv
RUN pipenv install -r requirements.txt

RUN mlflow server --backend-store-uri=sqlite:///mlflow.db --defaut-artifact-root=./mlruns
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "04-5_flask_app:app" ]




