FROM python:3.12-slim

WORKDIR /app

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow y dependencias para PostgreSQL y S3
RUN pip install --no-cache-dir mlflow psycopg2-binary boto3

EXPOSE 5000

CMD ["mlflow", "server"]