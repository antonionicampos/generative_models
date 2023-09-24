FROM tensorflow/tensorflow:2.13.0-gpu

LABEL maintainer "Antonioni Barros Campos <antonioni.campos@lps.ufrj.br>"

ADD requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt