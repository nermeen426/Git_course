# Use an official lightweight Linux distribution as the base image
FROM ubuntu:20.04

# Set environment variables to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sudo \
    python3 \
    python3-pip \
    build-essential \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | bash

# Copy the requirements file and install dependencies for FastAPI
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the entire application code
COPY . /app

# Expose ports for Ollama and FastAPI services
EXPOSE 11434 8000

# Start both Ollama and FastAPI services
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull tomasonjo/llama3-text2cypher-demo && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
