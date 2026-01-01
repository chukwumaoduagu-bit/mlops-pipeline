# Use official Python runtime
FROM python:3.12-slim

# Prevent Python bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including mlruns/ and mlflow.db)
COPY . .

# Expose port
EXPOSE 8000

# Run the API
CMD ["python", "src/api/app.py"]