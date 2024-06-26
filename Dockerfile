# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Install dependencies required for OpenCV and threading support
RUN apt-get update && apt-get install -y \
  libgl1-mesa-dev \
  libglib2.0-0

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 5050 available to the world outside this container
EXPOSE 5050

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
