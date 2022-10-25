FROM python:3.10

RUN apt update -y && apt upgrade -y
RUN pip install -U pip
RUN git clone https://github.com/s260895/mlops_bootcamp.git
RUN git init
WORKDIR /mlops_bootcamp
# COPY ["requirements.txt", "./"]
# COPY ["./data","./"]
# COPY ["./mlruns","mlflow.db","./"]
# COPY ["04-1_clean.py","04-2_train.py","04-3_predict.py","04-5_flask_app.py","./"]
RUN pip install -r requirements.txt

RUN useradd -u 5000 mlops
USER mlops:mlops
WORKDIR /app
RUN mlflow server --backend-store-uri=sqlite:///mlflow.db --defaut-artifact-root=./mlruns
EXPOSE 9696
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "04-5_flask_app:app" ]




