FROM continuumio/miniconda3

WORKDIR /app

COPY . /app
COPY environment.yml /app

RUN conda env create -f environment.yml