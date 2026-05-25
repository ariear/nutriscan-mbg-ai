# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install gdown to download models from Google Drive
RUN pip install --no-cache-dir gdown

# Download models from Google Drive
RUN gdown "https://drive.google.com/file/d/1FIB96vQQdrkQxddK9bluWFlIezLeK5jS/view?usp=sharing"

# Unzip the models into the models directory
RUN unzip models.zip -d models && rm models.zip

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Expose the port the app runs on
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
