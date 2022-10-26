FROM python:3.10
RUN apt update -y && apt upgrade -y
RUN pip install -U pip
# RUN git clone https://github.com/s260895/mlops_bootcamp.git
# RUN git init
WORKDIR /mlflow-server
COPY ["./mlruns","./"]
COPY ["mlflow.db","./"]
RUN pip install mlflow
EXPOSE 5000
ENTRYPOINT [ "mlflow", "server", "--backend-store-uri=sqlite:///mlflow.db", "--default-artifact-root=./mlruns", "--host=0.0.0.0"]