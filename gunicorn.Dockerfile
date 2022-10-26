FROM python:3.10
RUN apt update -y && apt upgrade -y
RUN pip install -U pip
# RUN git clone https://github.com/s260895/mlops_bootcamp.git
# RUN git init
WORKDIR /mlops_bootcamp
COPY ["04-1_clean.py", "./"]
COPY ["04-3_predict.py", "./"]
COPY ["04-5_flask_app.py", "./"]
COPY ["requirements.txt", "./"]
# COPY ["./data","./"]
RUN pip install -r requirements.txt 
EXPOSE 9696
ENTRYPOINT [ "python", "04-5_flask_app.py" ]
    # "gunicorn", "--bind=0.0.0.0:9696", "04-5_flask_app:app" ]