FROM python:3.6.7-alpine3.8

RUN apk --no-cache add gcc musl-dev

WORKDIR /app
COPY . /app

RUN pip install Cython==0.29
RUN pip install -r requirements.txt

CMD ["python", "main.py", "-u" , "data/domains.csv"]