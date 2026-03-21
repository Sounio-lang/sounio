FROM ubuntu:24.04

# Set working directory
WORKDIR /home/sounio

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create sounio user (optional, for non-root execution)
RUN useradd -m -s /bin/bash sounio || true

# Copy souc binary to /usr/local/bin/
COPY artifacts/omega/souc-bin/souc-linux-x86_64-jit /usr/local/bin/souc
RUN chmod +x /usr/local/bin/souc

# Copy stdlib to /opt/sounio/stdlib/
COPY stdlib /opt/sounio/stdlib

# Set environment variables
ENV SOUNIO_STDLIB_PATH=/opt/sounio/stdlib
ENV SOUC=/usr/local/bin/souc
ENV PYTHONUNBUFFERED=1

# Install Jupyter Lab
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir jupyter-lab jupyterlab ipykernel

# Install sounio-py (pure Python, from local path)
COPY triple-sounio-ecosystem/sounio-py /tmp/sounio-py
RUN cd /tmp/sounio-py && pip install --no-cache-dir -e . && rm -rf /tmp/sounio-py

# Install sounio-jupyter (from local path)
COPY triple-sounio-ecosystem/sounio-jupyter /tmp/sounio-jupyter
RUN cd /tmp/sounio-jupyter && pip install --no-cache-dir -e . && rm -rf /tmp/sounio-jupyter

# Install jupyter kernel spec for sounio
RUN python3.12 -m sounio_kernel.install

# Copy drug-discovery examples
COPY triple-sounio-ecosystem/drug-discovery/examples /home/sounio/examples

# Create work directory for notebooks
RUN mkdir -p /home/sounio/work && \
    chown -R sounio:sounio /home/sounio

# Copy jupyter config
COPY triple-sounio-ecosystem/docker/jupyter_config.py /root/.jupyter/jupyter_lab_config.py

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
