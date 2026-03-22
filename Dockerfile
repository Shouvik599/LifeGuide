# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Make the start script executable
RUN chmod +x start.sh

# HF Spaces requires port 7860
# We use the shell script as the entry point
CMD ["./start.sh"]