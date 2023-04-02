FROM supervisely/base-py-sdk:6.70.22
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y git

RUN pip3 install --upgrade pip
RUN git clone https://github.com/google-research/scenic.git /scenic
WORKDIR /scenic
RUN python3 -m pip install -vq .
WORKDIR /
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install ott-jax==0.2.0

RUN pip3 install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN git clone https://github.com/google-research/big_vision /big_vision
WORKDIR /big_vision
RUN pip3 install -r big_vision/requirements.txt
WORKDIR /

RUN python3 -m pip install supervisely==6.70.22