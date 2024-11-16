# Use an official Python image as the base
FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y wget gnupg2 ca-certificates \
    && apt-get clean

# Install Eclipse Temurin OpenJDK 8
RUN wget -O /tmp/temurin-8.tar.gz https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u372-b07/OpenJDK8U-jdk_x64_linux_hotspot_8u372b07.tar.gz \
    && mkdir -p /usr/local/java \
    && tar -xzf /tmp/temurin-8.tar.gz -C /usr/local/java --strip-components=1 \
    && rm /tmp/temurin-8.tar.gz

# Set Java 8 as the default version
ENV JAVA_HOME=/usr/local/java
ENV PATH="$JAVA_HOME/bin:$PATH"

# Verify Java installation
RUN java -version

# Set the working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=run.py
ENV FLASK_ENV=production

# Command to run the Flask application
CMD ["python", "run.py"]
