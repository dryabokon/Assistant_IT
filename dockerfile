FROM python:3.9-slim
WORKDIR /app
COPY . /app
ENV DEBIAN_FRONTEND=noninteractive 
COPY requirements.txt .
RUN apt-get update
RUN pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "main_webserver.py"]