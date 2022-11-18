FROM python:3.10
RUN apt update -y && apt upgrade -y
RUN pip install -U pip

WORKDIR /mldeployment-server

COPY ["04-1_clean.py", "./"]
COPY ["04-3_predict.py", "./"]
COPY ["04-5_flask_app.py", "./"]
COPY ["requirements.txt", "./"]

COPY ["./mlruns","./mlruns"]
COPY ["mlflow.db","./"]

RUN pip install -r requirements.txt 

EXPOSE 5000
EXPOSE 9696

CMD [ "mlflow", "server", "--backend-store-uri=sqlite:///mlflow.db", "--default-artifact-root=./mlruns", "--host=0.0.0.0"]
ENTRYPOINT [ "python", "04-5_flask_app.py" ]
    # "gunicorn", "--bind=0.0.0.0:9696", "04-5_flask_app:app" ]