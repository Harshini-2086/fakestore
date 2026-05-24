FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirments.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Cloud Run defaults to exposing port 8080
EXPOSE 8080

# Execute the application on boot
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
