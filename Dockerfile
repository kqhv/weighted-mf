FROM python:3.8-buster

RUN pip install \
        numpy==1.21.6 \
        pandas==1.3.5 \
        scipy==1.4.1 \
        scikit-learn==1.0.2 \
        torch==1.11.0

RUN pip install \
        pip install tqdm==4.64.0

WORKDIR root

RUN wget https://files.grouplens.org/datasets/movielens/ml-100k.zip >/dev/null 2>&1 && \
    unzip ml-100k.zip >/dev/null 2>&1

COPY *.py ./

RUN python main.py
