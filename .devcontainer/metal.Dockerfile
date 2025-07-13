FROM mcr.microsoft.com/devcontainers/base:jammy
RUN apt-get update && apt-get install -y curl && curl -sSL https://install.python-poetry.org | python3 - && ln -s $HOME/.local/bin/poetry /usr/local/bin/poetry
WORKDIR /workspace

