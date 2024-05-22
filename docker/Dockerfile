FROM ubuntu:20.04

# Install prerequisites
RUN apt-get update && apt-get install -y \
    curl \
    git \
    lsb-release \
    sudo \
    gnupg2 \
    build-essential \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Add your script to the Docker image
COPY install_rl_env.sh /root/install_rl_env.sh

# Make the script executable
RUN chmod +x /root/install_rl_env.sh

# Run the script in non-interactive mode
RUN /root/install_rl_env.sh -n

# Source environment variables
RUN echo "source /root/rl_ws/devel/setup.bash" >> /root/.bashrc

# Set entrypoint
ENTRYPOINT ["/bin/bash"]
