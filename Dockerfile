# debian bookworm
FROM debian:12

ENV DEBIAN_FRONTEND=noninteractive

# 1) Install dependencies
RUN apt update && apt install -y \
    build-essential \
    cmake \
    git \
    python3 python3-pip python3-venv \
    ocl-icd-opencl-dev \
    intel-opencl-icd \
    opencl-clhpp-headers \
    libx11-dev libjpeg-dev \
    python3-numpy python3-numpy-dev \
    libpocl2 \
 && rm -rf /var/lib/apt/lists/*

# 2) Create working directory and copy cadre-pse folder into it
WORKDIR /CADRE/cadre-pse
COPY ./cadre-pse/ .

# 3) Create and activate a Python virtual environment
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --upgrade pip matplotlib numpy

# 4) Install vim
RUN apt update && apt install -y vim

# 5) Symlink opencl.hpp to cl.hpp
RUN ln -s /usr/include/CL/opencl.hpp /usr/include/CL/cl.hpp

# 6) Install doctest
RUN git clone https://github.com/doctest/doctest.git /tmp/doctest && \
    cd /tmp/doctest && mkdir build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make install

# 7) Install dlib
RUN git clone https://github.com/davisking/dlib.git /tmp/dlib && \
cd /tmp/dlib && mkdir build && cd build && \
cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make install

# 8) Build simple-sim in /cadre-pse/simple-sim
WORKDIR /CADRE/cadre-pse/simple-sim
RUN mkdir build
WORKDIR /CADRE/cadre-pse/simple-sim/build
RUN cmake .. && make
