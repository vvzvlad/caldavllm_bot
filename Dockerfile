FROM python:3.11-slim

WORKDIR /app
RUN mkdir -p data && mkdir -p src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .
COPY src/*.py src/

CMD ["python", "main.py"]