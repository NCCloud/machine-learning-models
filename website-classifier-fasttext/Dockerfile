FROM python:3.6.7-slim

RUN apt-get update && apt-get -y install gcc g++

WORKDIR /app
COPY . /app

RUN pip install Cython==0.29
RUN pip install -r requirements.txt

CMD ["python", "main.py", "-u" , "data/domains.csv"]