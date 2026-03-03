FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY auth/ auth/

# Create keys directory (will be mounted as volume for persistence)
RUN mkdir -p /app/keys

EXPOSE 8001

CMD ["uvicorn", "auth.main:app", "--host", "0.0.0.0", "--port", "8001"]
