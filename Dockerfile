FROM python:3.10-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    wget \
    unzip \
    git \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libcanberra-gtk* \
    liblapack-dev \
    libx11-dev \
    libgl1-mesa-glx \
    espeak \
    python3-dev \
    libboost-all-dev \
    libboost-python-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY drowsiness_yawn.py .
COPY shape_predictor_68_face_landmarks.dat .
COPY haarcascade_frontalface_default.xml .

# Set display environment variable
ENV DISPLAY=:1

RUN chmod +x drowsiness_yawn.py

ENTRYPOINT ["python", "drowsiness_yawn.py"]
