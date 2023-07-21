FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt install python3-pip -y

RUN pip install "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN apt-get install -y git

RUN git clone https://github.com/google-research/scenic.git
WORKDIR /scenic
RUN git checkout 01b082706e7304e699baaa1bf4d15dcbf4d52de2
RUN python3 -m pip install -vq .
WORKDIR /

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install ml_collections
RUN pip3 install clu
RUN pip3 install tensorflow
RUN pip3 install tensorflow-datasets
RUN pip3 install scikit-image
RUN pip3 install bokeh

RUN python3 -m pip install supervisely==6.72.68

RUN mkdir -p variables
COPY variables_base_32.npy /variables/
COPY variables_base_16.npy /variables/
COPY variables_large_14.npy /variables/

COPY download.py /scenic/scenic/projects/baselines/clip/