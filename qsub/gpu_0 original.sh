#!/bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2401

/usr/local/bin/nvidia_entrypoint.sh
. /fefs/opt/dgx/env_set/nvcr-pytorch-2401.sh

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL
export PYTHONPATH="${HOME}/.raiden/nvcr-pytorch-2401/lib/python3.10/site-packages"

export PATH="${HOME}/.raiden/nvcr-pytorch-2401/bin:$PATH"
export LD_LIBRARY_PATH="${HOME}/.raiden/nvcr-pytorch-2401/lib:$LD_LIBRARY_PATH"
export LDFLAGS=-L/usr/local/nvidia/lib64
export PYTHONPATH="${HOME}/.raiden/nvcr-pytorch-2401/lib/python3.10/site-packages:${HOME}/vit_experiment"
export PYTHONUSERBASE="${HOME}/.raiden/nvcr-pytorch-2401"
export PREFIX="${HOME}/.raiden/nvcr-pytorch-2401"

#jupyter nbextension install --py widgetsnbextension --user
#jupyter nbextension enable widgetsnbextension --user --py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2678400 --allow-errors --debug 