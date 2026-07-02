# Use a slim Python 3.10 image
FROM python:3.10-slim

# Set environment variables to optimize Python execution in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory
WORKDIR /app

# Install curl for container health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK data to ensure offline capability and faster startup
RUN python -m nltk.downloader punkt punkt_tab stopwords

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 5000

# Run the app with gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "--workers", "2", "--timeout", "120"]
