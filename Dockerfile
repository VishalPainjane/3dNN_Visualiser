# All-in-One Dockerfile for NN3D Visualizer
# This Dockerfile builds both frontend and backend and serves them as a single container.

# --- Stage 1: Build Frontend ---
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package.json frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ ./

# Set environment variable for same-origin API
ENV VITE_API_URL=

# Build the frontend
RUN npm run build

# --- Stage 2: Backend and Runtime ---
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/backend/

# Copy built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Set environment variables
ENV DATABASE_PATH=/app/data/models.db
ENV FRONTEND_DIST_PATH=/app/frontend/dist
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create data directory
RUN mkdir -p /app/data

# Expose port 8000
EXPOSE 8000

# Start command
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
