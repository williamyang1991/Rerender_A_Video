### Builder image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# https://developer.nvidia.com/cuda-gpus
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX}"
ENV CLI_ARGS="${CLI_ARGS:-webUI.py}"
ENV CONTAINER_PORT="${CONTAINER_PORT:-7860}"
ENV USER_ID="${USER_ID:-1001}"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,rw \
  apt-get update && \
  apt-get install --no-install-recommends -y git vim build-essential python3-dev python3-venv cuda-nvcc-12-2 libglib2.0-0 libgl1-mesa-glx && \
  rm -rf /var/lib/apt/lists/*

# add user and drop root privileges
RUN useradd -m -u ${USER_ID} -d /home/app app && \
  mkdir -p /app && \
  chown -R app:app /app
USER app

# Clone repo abd submodules
RUN git clone --depth=1 https://github.com/williamyang1991/Rerender_A_Video.git --recursive /app

WORKDIR /app

# Install virtualenv and deps
RUN --mount=type=cache,target=/home/app/.cache/pip,rw \
  python3 -m venv venv && \
  . venv/bin/activate && \
  echo "git tag: $(git tag) commit: $(git rev-parse HEAD)" > /app/version.txt && \
  pip3 install --upgrade pip setuptools wheel && \
  pip3 install -r requirements.txt

# Expose and run
VOLUME [ "/app/models", "/app/result", "/home/app/.cache" ]
EXPOSE ${CONTAINER_PORT}
CMD ["/app/entrypoint.sh"]
