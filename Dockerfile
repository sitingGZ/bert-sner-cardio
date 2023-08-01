FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3.9 python3-distutils python3-pip python3-apt
RUN apt-get install nano
WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY ./app /usr/src/app
CMD gunicorn app:server -b :8050