FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
RUN apt-get -o Acquire::Retries=3 update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get -o Acquire::Retries=3 update \
 && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        curl git build-essential ca-certificates jq \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# pip (optional: use `apt-get install python3-pip` instead)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

ENV POETRY_HOME=/opt/poetry PATH="$POETRY_HOME/bin:$PATH"
RUN curl -sSL https://install.python-poetry.org | python3 -

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