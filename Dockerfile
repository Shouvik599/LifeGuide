# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Run the data ingestion script
RUN python ingest.py

# Run the application in Hugging Face Space
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
