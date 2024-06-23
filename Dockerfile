# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire current directory contents into the container at /app
COPY . .

# Install dependencies
RUN pip install torch torchvision pillow gradio

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# CMD to specify the command to run your application
CMD ["python", "app.py"]