# =====================================================================
# FINAL & STABLE DOCKERFILE
# =====================================================================

# Stage 1: Base image with Python and Poetry
# Using 'AS' in uppercase to resolve build warnings.
FROM python:3.11-slim AS base

# Set environment variables for Python and Pip timeout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION="1.8.2" \
    # This is the new, robust way to handle timeouts. Sets a 5-minute timeout for pip.
    PIP_DEFAULT_TIMEOUT=300

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using pip for a more deterministic installation
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Add Poetry to the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock ./

# --------------------------------------------------------------------

# Stage 2: Production image
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Copy Poetry and application files
COPY --from=base /root/.local/bin/poetry /usr/local/bin/poetry
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install only production dependencies (will use the timeout from the base image)
RUN poetry install --no-dev --no-interaction --no-ansi

COPY --chown=appuser:appuser ./src /app/src
RUN chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------

# Stage 3: Development image
FROM base AS development

# Install all dependencies (will use the timeout from the base image)
RUN poetry install --no-interaction --no-ansi

COPY ./src /app/src
COPY ./tests /app/tests

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]