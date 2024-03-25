FROM continuumio/miniconda3

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY environment.yml .
RUN conda env create -f environment.yml
RUN echo "conda activate m2" >> ~/.bashrc

SHELL ["conda", "run", "-n", "m2", "/bin/bash", "-c"]
