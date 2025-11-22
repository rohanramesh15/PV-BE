# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for PIL, numpy, and other packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port (Google Cloud Run uses PORT environment variable)
ENV PORT=8080
EXPOSE 8080

# Use gunicorn to run the Flask app in production
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
