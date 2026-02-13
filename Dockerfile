# Sounio Compiler Docker Image
# =============================
# Reproducible build environment for the Sounio programming language.
#
# DOI: 10.5281/zenodo.18404188
# Version: 0.100.0
# License: MIT
#
# Usage:
#   Build:   docker build -t sounio:0.100.0 .
#   Run:     docker run -v $(pwd):/workspace sounio:0.100.0 check /workspace/myfile.sio
#   Tests:   docker build --target test -t sounio:test .
#   Shell:   docker run -it -v $(pwd):/workspace --entrypoint /bin/bash sounio:0.100.0
#   REPL:    docker run -it sounio:0.100.0 repl
#
# For scientific reproducibility, use the specific version tag:
#   docker pull ghcr.io/sounio-lang/sounio:0.100.0

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.92-slim AS builder

# Build metadata for reproducibility
LABEL org.opencontainers.image.title="Sounio Compiler"
LABEL org.opencontainers.image.version="0.100.0"
LABEL org.opencontainers.image.description="Systems programming language for epistemic computing"
LABEL org.opencontainers.image.authors="Demetrios Chiuratto Agourakis"
LABEL org.opencontainers.image.url="https://souniolang.org"
LABEL org.opencontainers.image.source="https://github.com/Sounio-lang/sounio"
LABEL org.opencontainers.image.documentation="https://souniolang.org/docs/"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.vendor="Sounio Project"
LABEL org.label-schema.schema-version="1.0"
LABEL science.doi="10.5281/zenodo.18404188"

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace sources needed to build `souc`
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY stdlib/ ./stdlib/
COPY examples/ ./examples/
COPY benches/ ./benches/

# Build release binary with common features
RUN cargo build --release -p souc --features "jit,pkg"

# =============================================================================
# Stage 2: Test (optional - build with --target test)
# =============================================================================
FROM builder AS test

# Run tests to verify build integrity
RUN cargo test --release -p souc --features "jit,pkg" -- --test-threads=1

# =============================================================================
# Stage 3: Runtime
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Runtime metadata
LABEL org.opencontainers.image.title="Sounio Compiler"
LABEL org.opencontainers.image.version="0.100.0"
LABEL org.opencontainers.image.description="Systems programming language for epistemic computing"
LABEL org.opencontainers.image.authors="Demetrios Chiuratto Agourakis"
LABEL org.opencontainers.image.url="https://souniolang.org"
LABEL org.opencontainers.image.licenses="MIT"
LABEL science.doi="10.5281/zenodo.18404188"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/target/release/souc /usr/local/bin/souc

# Copy standard library
COPY --from=builder /build/stdlib /usr/local/share/sounio/stdlib

# Set environment for stdlib resolution
ENV SOUNIO_STDLIB_PATH=/usr/local/share/sounio/stdlib

# Set up working directory
WORKDIR /workspace

# Verify installation
RUN souc --version

# Default command
ENTRYPOINT ["souc"]
CMD ["--help"]
