# Use an official Ubuntu base image
FROM ubuntu:20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    zip \
    pkg-config \
    curl \
    tar \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

# Add vcpkg to PATH
ENV PATH="/opt/vcpkg:${PATH}"

# Set the working directory
WORKDIR /project

# Copy only the vcpkg.json first to leverage Docker caching
COPY vcpkg.json /project/vcpkg.json

# Install vcpkg dependencies as per vcpkg.json
RUN /opt/vcpkg/vcpkg install

# Optional: Integrate vcpkg (useful for CMake to find packages automatically)
RUN /opt/vcpkg/vcpkg integrate install

RUN apt-get install -y \
    cmake
# Set entrypoint to bash for interactive use (optional)
#ENTRYPOINT ["/bin/bash"]
