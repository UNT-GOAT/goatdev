# Goat Grading API Dockerfile
# Build from repo root: docker build -f goat-api/Dockerfile -t goat-api .

FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY goat-api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY goat-api/api/ ./api/

# Copy model files (only .pt and calibration .json files)
COPY model/side/best.pt ./model/side/
COPY model/side/side_calibration.json ./model/side/
COPY model/top/best.pt ./model/top/
COPY model/top/top_calibration.json ./model/top/
COPY model/front/best.pt ./model/front/
COPY model/front/front_calibration.json ./model/front/

# Create data and debug directories
RUN mkdir -p /app/data /app/data/debug

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
