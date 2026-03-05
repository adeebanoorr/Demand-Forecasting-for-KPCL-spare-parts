# Stage 1: Build React
FROM node:18-slim AS frontend-builder
WORKDIR /app/src/webapp
COPY src/webapp/package*.json ./
RUN npm install
COPY src/webapp/ ./
RUN npm run build

# Stage 2: Run Python/FastAPI
FROM python:3.12-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (will follow .gitignore, excluding large models)
COPY . .

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/src/webapp/dist ./src/webapp/dist

# The Port uvicorn will run on (Railway provides $PORT)
ENV PORT=8000
EXPOSE 8000

# Start the application using Gunicorn for production robustness
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "src.api.main:app", "--bind", "0.0.0.0:8000"]
