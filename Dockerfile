# Use an official lightweight Python image
FROM python:3.9

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY app /app/app/
COPY weights /app/weights/
COPY input.png /app/input.png

# Copy the Ultralytics repo without installing via pip
COPY ultralytics /app/ultralytics/

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# ✅ Set PYTHONPATH to recognize both /app and /app/ultralytics
ENV PYTHONPATH="/app:/app/ultralytics:$PYTHONPATH"

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV and PaddleOCR
RUN pip install --no-cache-dir opencv-python-headless paddleocr paddlepaddle

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
