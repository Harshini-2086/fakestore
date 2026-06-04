
FROM python:3.11-slim

WORKDIR /app

# Install system-level dependencies if needed (libpq is the postgres C library)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Cloud Run defaults to exposing port 8080
EXPOSE 8080

# Environment variables to optimize Python inside Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Execute the application on boot
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]