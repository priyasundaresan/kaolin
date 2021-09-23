#FROM nvidia/cuda:10.2-base-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libhdf5-dev \
      libglib2.0-0 \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim wget \
    build-essential cmake \
    libopenblas-dev \
    libboost-dev \
    libboost-all-dev \
    mesa-common-dev \
    libglu1-mesa-dev \
    freeglut3-dev

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

RUN pip install --upgrade pip 

RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    gcc-4.8 gcc-5 \
    g++-4.8 g++-5 \
    libblas-dev liblapack-dev 

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 20
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 20

RUN apt update && apt install -y --no-install-recommends \
    scons \
    libpng-dev 

# conda stuff
RUN apt-get update \
    && apt-get install -y wget \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b

ADD kaolin.yml /tmp/kaolin.yml
RUN conda env create -f /tmp/kaolin.yml
RUN echo "conda activate kaolin" >> ~/.bashrc
SHELL ["conda", "run", "-n", "kaolin", "/bin/bash", "-c"]
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin /tmp/kaolin 
WORKDIR /tmp/kaolin
RUN python /tmp/kaolin/setup.py develop

WORKDIR /host
CMD ["/bin/bash"]
