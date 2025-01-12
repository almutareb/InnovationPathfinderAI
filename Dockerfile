# Use a lightweight Python 3.11 base image
FROM python:3.11-slim-buster

# Expose the application port
EXPOSE 8000

# Set the working directory
WORKDIR /app

# Install necessary system packages
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     ffmpeg \
#     sqlite3 \
#     curl \
#     gnupg \
#     g++ \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY requirements.txt /app/
COPY . /app/


# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --upgrade -r requirements.txt

# For running the container locally 
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["python", "app.py"]