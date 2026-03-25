FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user for HF compliance
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory to user's home
WORKDIR $HOME/app

# Copy application code and set ownership to our user
COPY --chown=user . $HOME/app

# Ensure the start script is executable
RUN chmod +x start.sh

# HF Spaces requires port 7860
EXPOSE 7860

CMD ["./start.sh"]