# Sounio Compiler Docker Image
#
# Build: docker build -t sounio .
# Run:   docker run -v $(pwd):/workspace sounio check /workspace/myfile.sio
# Shell: docker run -it -v $(pwd):/workspace sounio /bin/bash

FROM rust:1.84-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source files
COPY compiler/ ./compiler/
COPY stdlib/ ./stdlib/
COPY examples/ ./examples/

# Build release binary with common features
WORKDIR /build/compiler
RUN cargo build --release --features "jit,pkg"

# Runtime image
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/compiler/target/release/souc /usr/local/bin/souc

# Copy standard library
COPY --from=builder /build/stdlib /usr/local/share/sounio/stdlib

# Set up working directory
WORKDIR /workspace

# Default command
ENTRYPOINT ["souc"]
CMD ["--help"]
