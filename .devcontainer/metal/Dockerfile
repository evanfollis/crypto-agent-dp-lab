FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC
RUN apt-get -o Acquire::Retries=3 update \
 && apt-get install -y --no-install-recommends software-properties-common wget \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get -o Acquire::Retries=3 update \
 && apt-get install -y --no-install-recommends \
        curl git build-essential ca-certificates jq \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
 && rm -rf /var/lib/apt/lists/*

# Build Python 3.11.10 from source to ensure we get a stable version
RUN cd /tmp \
 && wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tgz \
 && tar -xzf Python-3.11.10.tgz \
 && cd Python-3.11.10 \
 && ./configure --enable-optimizations --with-ensurepip=install \
 && make -j$(nproc) \
 && make altinstall \
 && cd / \
 && rm -rf /tmp/Python-3.11.10*

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.11 1

# pip is already installed with Python 3.11.10 via --with-ensurepip
# Just ensure pip is up to date
RUN python3.11 -m pip install --upgrade pip

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

