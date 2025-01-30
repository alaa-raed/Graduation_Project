FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    libopencv-dev libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    libgtk2.0-dev \
    libsm6 libxext6 libxrender-dev \
    espeak \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY drowsiness_yawn.py .
COPY haarcascade_frontalface_default.xml .
COPY shape_predictor_68_face_landmarks.dat .

# Set display environment variable
ENV DISPLAY=:0

ENTRYPOINT ["python", "drowsiness_yawn.py"]
