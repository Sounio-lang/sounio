FROM scratch

ARG RELEASE_TAG
ARG SOURCE_COMMIT
LABEL org.opencontainers.image.source="https://github.com/Sounio-lang/sounio" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}" \
      org.opencontainers.image.version="${RELEASE_TAG}" \
      org.opencontainers.image.description="Pinned Madaros compiler and stdlib archive; extract before use"

# This is a distribution archive package, not a runnable compiler image.
COPY madaros-distribution.tar.gz /distribution/madaros-distribution.tar.gz
COPY SHA256SUMS /distribution/SHA256SUMS
