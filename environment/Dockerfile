FROM public.ecr.aws/docker/library/ubuntu:24.04@sha256:0d39fcc8335d6d74d5502f6df2d30119ff4790ebbb60b364818d5112d9e3e932
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends git python3 python3-pip bash curl wget python3-venv python3-dev \
    && python3 -m pip install --no-cache-dir pytest==8.4.1 \
    && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN mkdir -p /logs

WORKDIR /app

# PyTorch is the public base repository for this task. The exact commit is pinned for reproducibility.
RUN git init && git remote add origin https://github.com/pytorch/pytorch.git \
    && git fetch --depth 1 origin 990e00ebd48fdd1d55793be6110f479c8a29244e \
    && git checkout FETCH_HEAD && rm -rf .git

# Task-specific public input; no solution or verifier ground truth is copied into the image.
COPY codebase/ .

COPY gitignore /tmp/harbor.gitignore
RUN cat /tmp/harbor.gitignore >> .gitignore && rm /tmp/harbor.gitignore

RUN git config --global user.email "eval@bench"; \
    git config --global user.name "Eval Runner"; \
    git init; \
    git add -A; \
    git commit -m "initialize repo" || true; \
    git tag v0.1.0-init
CMD ["bash"]
