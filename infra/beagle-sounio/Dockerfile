# beagle-sounio — cluster runtime image for Sounio campaigns
# ============================================================
# Purpose: a minimal, reproducible image for running Sounio .sio programs
# and their Python companions on the BeagleCockpit-managed k8s cluster.
# Consumed by cockpit_submit_job calls; not intended for local development.
#
# Build:
#   cd /workspace/sounio
#   docker build -f infra/beagle-sounio/Dockerfile \
#       -t ttl.sh/beagle-sounio-$(date +%s):24h .
#   docker push ttl.sh/beagle-sounio-<tag>:24h
#
# Pass the resulting tag as `image:` to cockpit_submit_job.
#
# Layout inside the image:
#   /app/                               repo subset (ROOT for Python scripts)
#     bin/souc                          wrapper (on PATH)
#     artifacts/self-hosted/souc-...    self-hosted compiler ELF
#     stdlib/                           Sounio standard library
#     scripts/research/                 ABIDE preprocessor + helpers
#     experiments/                      campaign code (baked at build time)
#   /orangefs/                          OrangeFS mount point (k8s volume)
#   /workspace/                         user-writable scratch (WORKDIR)
#
# Entrypoint: symlinks /app/artifacts/research/abide → OrangeFS abide-data so
# unmodified Python scripts write their heavy outputs straight to OrangeFS.
#
# Non-root: runs as UID 1000 (agent). COPY --chown ensures /app is writable.
#
# Image budget: ~300 MB (python:3.11-slim 80 MB + Python deps ~200 MB +
# Sounio compiler + stdlib ~17 MB). Do NOT bake torch/jax here — that's a
# separate beagle-sounio-gpu image.

# syntax=docker/dockerfile:1.7

FROM python:3.11-slim AS base

# ---------------------------------------------------------------------------
# System setup: CA certs (HTTPS for S3 / ABIDE), tini (PID 1 reaper),
# mount utilities (for entrypoint's mountpoint check).
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        tini \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged agent user. UID 1000 matches the typical workspace convention
# so OrangeFS permissions line up (adjust via `--build-arg AGENT_UID=N` if
# your cluster uses a different convention).
ARG AGENT_UID=1000
ARG AGENT_GID=1000
RUN groupadd -g "${AGENT_GID}" agent && \
    useradd -u "${AGENT_UID}" -g "${AGENT_GID}" -m -s /bin/bash agent

# ---------------------------------------------------------------------------
# Python deps. Installed as root into site-packages, usable by any user.
# ---------------------------------------------------------------------------
COPY infra/beagle-sounio/requirements.txt /opt/beagle/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /opt/beagle/requirements.txt

# ---------------------------------------------------------------------------
# Sounio runtime. Targeted COPYs so the image stays lean even though the
# build context is the full repo.
# ---------------------------------------------------------------------------
COPY --chown=agent:agent bin/souc \
     /app/bin/souc
COPY --chown=agent:agent artifacts/self-hosted/souc-self-hosted-x86_64 \
     /app/artifacts/self-hosted/souc-self-hosted-x86_64
COPY --chown=agent:agent stdlib/ \
     /app/stdlib/
COPY --chown=agent:agent scripts/research/ \
     /app/scripts/research/
COPY --chown=agent:agent experiments/ \
     /app/experiments/

# bin/souc is a bash wrapper that execs the ELF; ensure it's executable.
RUN chmod +x /app/bin/souc /app/artifacts/self-hosted/souc-self-hosted-x86_64

# ---------------------------------------------------------------------------
# Entrypoint: wires OrangeFS into /app/artifacts/research/abide at runtime.
# ---------------------------------------------------------------------------
COPY infra/beagle-sounio/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ---------------------------------------------------------------------------
# Mount points and env.
# ---------------------------------------------------------------------------
RUN mkdir -p /orangefs /workspace && \
    chown agent:agent /orangefs /workspace

ENV PATH="/app/bin:${PATH}" \
    SOUC_BIN=/app/bin/souc \
    SOUNIO_STDLIB_PATH=/app/stdlib \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OCI labels — cockpit_submit_job can read these for provenance tracking.
LABEL org.opencontainers.image.title="beagle-sounio" \
      org.opencontainers.image.description="Sounio cluster runtime for BeagleCockpit campaigns" \
      org.opencontainers.image.vendor="Sounio Project" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/Sounio-lang/sounio" \
      beagle.sounio.role="campaign-runtime" \
      beagle.sounio.compiler="self-hosted-x86_64"

USER agent
WORKDIR /workspace

# tini reaps zombies (important for long-running pilots that fork subprocesses
# like the ABIDE preprocessor). entrypoint.sh then wires OrangeFS and execs
# the caller's command.
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint.sh"]

# Default: print souc info so `docker run <image>` without args is a sanity
# check, not a silent no-op.
CMD ["bash", "-c", "souc info && python -c 'import numpy, scipy, pandas, matplotlib; print(\"python deps OK\")'"]
