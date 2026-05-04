ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG TARGETPLATFORM
ARG BUILDPLATFORM

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_INPUT=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN pip install --upgrade pip wheel setuptools

RUN pip install \
    "tensorflow>=2.19.0,<3.0.0" \
    "dlib>=19.24.8,<20.0.0" \
    "opencv-python-headless>=4.11.0.86,<5.0.0.0" \
    "pyyaml>=6.0.3,<7.0.0" \
    "tqdm>=4.66.0,<5.0.0"

WORKDIR /weights-stage
COPY --from=weights / ./
RUN unzip -oq lm_feat.zip && \
    unzip -oq weights.zip && \
    rm -f lm_feat.zip weights.zip

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime
ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    liblapack3 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk-3-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /usr/local/lib/python${PYTHON_VERSION}/site-packages \
                    /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=app:app bin_video_file.py ./bin_video_file.py
COPY --chown=app:app displayers ./displayers
COPY --chown=app:app model_managers ./model_managers
COPY --chown=app:app tf_models ./tf_models
COPY --chown=app:app utils ./utils

COPY --from=builder --chown=app:app /weights-stage/lm_feat ./lm_feat
COPY --from=builder --chown=app:app /weights-stage/weights ./weights

RUN mkdir -p /tmp/gaze-data && chown app:app /tmp/gaze-data

USER app

ENV TF_CPP_MIN_LOG_LEVEL=2 \
    XDG_CACHE_HOME=/tmp/gaze-data \
    HOME=/tmp/gaze-data

ENTRYPOINT ["python", "bin_video_file.py"]
CMD ["--help"]
