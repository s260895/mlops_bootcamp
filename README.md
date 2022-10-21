# mlops_bootcamp

# on base machine install docker
sudo apt install docker.io

# add current user to docker group
sudo usermod -aG docker $USER
# logout and log back in after this

# check if docker running correctly
docker run hello-world

# create and run ubuntu container named 'mlops_instance' and with port 5000 exposed 
docker run -it --name=mlops_instance -p 5000:5000 ubuntu bash

# once inside container, update & upgrade apt repositories
apt update
apt upgrade

# install sudo (limited super user privileges)
apt install sudo

# create mlops user with specified home directory
sudo useradd -m -d /home/mlops mlops
# set password for mlops user
passwd dev

# add dev user to sudoers
sudo adduser mlops sudo

# logout of mlops_instance
exit
# this will also stop the instance

# check if mlops_instance was created
docker container ls -a

# start mlops_instance
docker container start mlops_instance

# login to mlops_instance
docker exec -it --user=mlops mlops_instance bash

# install wget & curl(file downloading tool)
sudo apt install wget
sudo apt install curl


# install vim (file editing tool)
sudo apt install vim

# install git
sudo apt install git

# go to home directory
cd ~

# download and install anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash Anaconda3-2022.05-Linux-x86_64.sh

# download and install vscode-server
wget https://aka.ms/install-vscode-server/setup.sh 
bash setup.sh


# lgout and log back in
exit
docker exec -it ...

# check if anaconda installed correctly
which conda
which anaconda
which python
# all three should point to a directory in /home/$USER/anaconda3/bin/...

# make directory to store git repos 
mkdir git
cd git

# make mlops bootcamp project directory
mkdir mlops_bootcamp
cd mlops_bootcamp

# at root level of project create requirements.txt file
touch requirements.txt

# verify if requirements.txt created
ls

# open requirements.txt in vim
vim requirements.txt

# once opened, enter edit mode by pressing i

# put list of required libraries inside requirements.txt (each new project goes on newline) 
mlflow
jupyter
scikit-learn
pandas
seaborn
hyperopt
xgboost

# save changes by escaping edit mode using esc, and pressing wq and enter to write changes and quit vim.

# verify if changes saved
cat requirements.txt

# create new conda environment 
conda create -n mlops_bootcamp_env python=3.9

# check if environment created 
conda info --envs

# spin up vscode server
# this opens up a url to remote vscode environment, do all following steps in that env
# runs on port 34547
code-server

# activate created conda environment
conda activate mlops_bootcamp_env

# install requirements using pip
pip install -r requirements.txt

# verify if libraries are installed
pip list

# start mlflow ui with sqlite backend
# runs on port 5000
mlflow ui --backend-store-uri=sqlite:///mlflow.db

# start prefect orion workflow orchestration
# runs on port 4200
prefect orion start

# create prefect orion deployment using py file
prefect deployment create <name-of-file.py>

