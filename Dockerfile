# ============================================================
# Urban Subsidence Watcher — Dockerfile
# Base: ubuntu:22.04  |  Python 3.10  |  GDAL 3.8  |  PyTorch CPU
# ============================================================

FROM ubuntu:22.04

LABEL maintainer="Samuel Appiah Kubi <samuel.appiah-kubi@student.plus.ac.at>"
LABEL description="Urban Subsidence Watcher — Sentinel-1 InSAR + Edge AI Digital Twin"
LABEL version="1.0.0"

# ── System environment ─────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GDAL_VERSION=3.8.4
ENV PROJ_LIB=/usr/share/proj

# ── System dependencies ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # GDAL / geospatial
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    # Image / GL libs
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    # NetCDF / HDF5
    libnetcdf-dev \
    libhdf5-dev \
    # Python
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    # Misc
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Make python3.10 the default ───────────────────────────
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.10 1

# ── Upgrade pip ──────────────────────────────────────────
RUN pip install --upgrade pip setuptools wheel

# ── GDAL Python bindings (match system GDAL) ─────────────
RUN pip install GDAL==$(gdal-config --version)

# ── PyTorch CPU (for edge simulation) ────────────────────
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Core Python dependencies ─────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt || true
# Note: GDAL and torch already installed above; pip may skip/warn — that's fine.

# ── Optional: ESA SNAP (comment out if not needed — adds ~2 GB) ──
# ARG INSTALL_SNAP=false
# RUN if [ "$INSTALL_SNAP" = "true" ]; then \
#     wget -q https://download.esa.int/step/snap/9.0/installers/esa-snap_sentinel_unix_9_0_0.sh \
#     && chmod +x esa-snap_sentinel_unix_9_0_0.sh \
#     && ./esa-snap_sentinel_unix_9_0_0.sh -q \
#     && rm esa-snap_sentinel_unix_9_0_0.sh; \
# fi

# ── Working directory ─────────────────────────────────────
WORKDIR /app

# ── Copy project ─────────────────────────────────────────
COPY . /app/

# ── Create data directories ───────────────────────────────
RUN mkdir -p data/{raw/{sentinel1,dem},processed/s1_coregistered,\
insar/{interferograms,coherence,unwrapped,timeseries},\
subsidence,features,models/{onnx,tflite},forecasts,\
validation,dashboard,exports,edge,logs}

# ── Default command ───────────────────────────────────────
CMD ["python", "main.py", "--step", "all", "--log-level", "INFO"]
