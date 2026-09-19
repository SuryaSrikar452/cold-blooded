# SYNTHIA Clinical Synthetic Data Platform - Production Dockerfile for Render
FROM python:3.11-slim

# Install system dependencies & Node.js 20
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js backend dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm install --production

# Copy application files, data models, and assets
COPY . .

# Set environment variables for Render
ENV PORT=5000
ENV NODE_ENV=production

EXPOSE 5000

# Start Express server (serves frontend SPA and interfaces with Python synthetic engine)
CMD ["node", "backend/server.js"]
