FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for build and wfdb/neurokit
RUN apt-get update && apt-get install -y gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY backend /app/backend
COPY CardioVision_Feature_Pipeline /app/CardioVision_Feature_Pipeline

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
