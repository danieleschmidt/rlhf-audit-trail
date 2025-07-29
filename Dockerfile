# Multi-stage Docker build for RLHF Audit Trail
# Optimized for production use with security best practices

# Build stage
FROM python:3.10-slim-bullseye as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.authors="Daniel Schmidt"
LABEL org.opencontainers.image.url="https://github.com/terragonlabs/rlhf-audit-trail"
LABEL org.opencontainers.image.documentation="https://rlhf-audit-trail.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/rlhf-audit-trail"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.title="RLHF Audit Trail"
LABEL org.opencontainers.image.description="End-to-end pipeline for verifiable provenance of RLHF steps with EU AI Act compliance"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.10-slim-bullseye as production

# Set build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add same metadata labels as builder
LABEL maintainer="Daniel Schmidt <daniel@terragonlabs.com>"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.authors="Daniel Schmidt"
LABEL org.opencontainers.image.url="https://github.com/terragonlabs/rlhf-audit-trail"
LABEL org.opencontainers.image.documentation="https://rlhf-audit-trail.readthedocs.io/"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/rlhf-audit-trail"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.title="RLHF Audit Trail"
LABEL org.opencontainers.image.description="End-to-end pipeline for verifiable provenance of RLHF steps with EU AI Act compliance"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set up application directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/audit_logs /app/checkpoints \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import rlhf_audit_trail; print('OK')" || exit 1

# Expose ports
EXPOSE 8501 8000

# Default command
CMD ["python", "-m", "rlhf_audit_trail.dashboard.app"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Set development environment variables
ENV ENVIRONMENT=development
ENV DEBUG=1
ENV LOG_LEVEL=DEBUG

# Create development user
RUN groupadd --gid 1000 devuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash devuser \
    && usermod -aG sudo devuser

# Set up development workspace
WORKDIR /workspace
COPY --chown=devuser:devuser . .

# Switch to development user
USER devuser

# Install pre-commit hooks
RUN pre-commit install

# Development command
CMD ["bash"]