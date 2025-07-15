FROM ubuntu:jammy

# Set timezone to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.11.8 and essential tools
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3.11=3.11.8-1+jammy1 \
        python3.11-dev=3.11.8-1+jammy1 \
        python3.11-venv=3.11.8-1+jammy1 \
        python3.11-distutils=3.11.8-1+jammy1 \
        curl \
        git \
        build-essential \
        ca-certificates \
        jq \
 && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2

# Set PATH for user-installed binaries
ENV PATH="/home/developer/.local/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Create non-root user
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Switch to non-root user
USER $USERNAME

