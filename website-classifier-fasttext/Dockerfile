FROM python:3.6.7-slim

RUN apt-get update && apt-get -y install gcc g++ git

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

RUN git clone https://github.com/facebookresearch/fastText.git && \
    cd fastText && pip install .

CMD ["python", "main.py", "-u" , "data/domains.csv"]