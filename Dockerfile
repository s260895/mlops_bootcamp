FROM ubuntu:jammy

LABEL maintainer='Sumeet Zankar'
LABEL version="0.1"
LABEL description="MLOps Bootcamp Image for development"

# set user as root
USER root

# update and upgrade apt repositories
RUN apt-get -y update && apt-get -y upgrade 

# install wget,vim and git
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y git

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
RUN bash Anaconda3-2022.05-Linux-x86_64.sh -b
RUN rm -r Anaconda3-2022.05-Linux-x86_64.sh

RUN ls -a ~
RUN pwd ~

RUN echo 'export PATH=/root/anaconda3/bin:$PATH' >> ~/.bashrc

WORKDIR /root/git/mlops_bootcamp

COPY ./requirements.txt /root/git/mlops_bootcamp/
RUN conda create -n mlops_bootcamp_env python=3.9
RUN conda activate mlops_bootcamp_env
RUN pip install -r requirements.txt
ENTRYPOINT mlflow ui --backend-store-uri=sqlite:///mlflow.db

EXPOSE 5000
