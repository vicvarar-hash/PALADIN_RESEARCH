FROM python:3.12-slim
WORKDIR /artifact
COPY requirements-replay.txt .
RUN pip install --no-cache-dir -r requirements-replay.txt
COPY . .
CMD ["python", "reproduce/run.py"]
