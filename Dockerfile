FROM tensorflow/tensorflow:2.13.0-gpu

LABEL maintainer "Antonioni Barros Campos <antonioni.campos@lps.ufrj.br>"
LABEL version "0.0.1"

WORKDIR /generative_models

COPY requirements.txt .
COPY /main ./main
COPY /wgan ./wgan

RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /generative_models/main

RUN python3 download_mnist.py