# bert-sner-cardio
BERT-SNER for Cardiology

Build Image: docker build -t sner-app .

HOST = 
Run Docker Container with the created Image and map to the given HOST:  docker run -dp $HOST:8050:8050 sner-app