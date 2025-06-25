FROM python:3.11-slim

WORKDIR /code

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p static/uploads

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production

# Optional: Install system dependencies for image processing
# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "main:app"] 