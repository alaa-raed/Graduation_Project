FROM python:3.11-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    espeak \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY drowsiness_detection.py .
COPY haarcascade_frontalface_default.xml .
COPY shape_predictor_68_face_landmarks.dat .

# Set display environment variable
ENV DISPLAY=:0

ENTRYPOINT ["python", "drowsiness_detection.py"]
