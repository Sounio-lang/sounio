FROM ubuntu:24.04

# Set working directory
WORKDIR /home/sounio

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    git \
    curl \
    jq \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create sounio user (optional, for non-root execution)
RUN useradd -m -s /bin/bash sounio || true

# Preview compiler distribution supports Linux x86_64.
RUN test "$(uname -m)" = x86_64
ENV SOUC=/opt/sounio/bin/souc
ENV PYTHONUNBUFFERED=1

# Install published, checksum-pinned Python/Jupyter wheels into an owned venv.
# The fetcher needs only the release catalog, not either in-tree package.
COPY ecosystem/external-releases.json /opt/sounio-consumer/ecosystem/external-releases.json
COPY scripts/ci/fetch_external_ecosystem.py /opt/sounio-consumer/scripts/ci/fetch_external_ecosystem.py
RUN python3.12 /opt/sounio-consumer/scripts/ci/fetch_external_ecosystem.py /tmp/sounio-compiler --suffix .tar.gz && \
    mkdir -p /opt/sounio && \
    tar -xzf /tmp/sounio-compiler/madaros-distribution.tar.gz --strip-components=1 -C /opt/sounio && \
    python3.12 /opt/sounio/verify-distribution.py --verify /opt/sounio && \
    rm -rf /tmp/sounio-compiler
ENV PATH="/opt/sounio/bin:${PATH}"

RUN python3.12 -m venv /opt/sounio-python && \
    python3.12 /opt/sounio-consumer/scripts/ci/fetch_external_ecosystem.py /tmp/sounio-wheels --suffix .whl && \
    /opt/sounio-python/bin/python -m pip install --no-cache-dir jupyterlab /tmp/sounio-wheels/*.whl && \
    rm -rf /tmp/sounio-wheels
ENV PATH="/opt/sounio-python/bin:${PATH}"

# Install jupyter kernel spec for sounio
RUN python -c "import json, sys; from pathlib import Path; from sounio_kernel import kernelspec; p = Path(sys.prefix) / 'share/jupyter/kernels/sounio'; p.mkdir(parents=True, exist_ok=True); (p / 'kernel.json').write_text(json.dumps(kernelspec()))"

# Copy drug-discovery examples
COPY ecosystem/drug-discovery/examples /home/sounio/examples

# Create work directory for notebooks
RUN mkdir -p /home/sounio/work && \
    chown -R sounio:sounio /home/sounio

# Copy jupyter config
COPY ecosystem/docker/jupyter_config.py /root/.jupyter/jupyter_lab_config.py

# Expose Jupyter port
EXPOSE 8888

# Set entrypoint
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
