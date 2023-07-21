FROM supervisely/base-py-sdk:6.70.22
ENV DEBIAN_FRONTEND=noninteractive
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN apt-get update
RUN apt-get install -y git

RUN pip3 install --upgrade pip
RUN git clone https://github.com/google-research/scenic.git
WORKDIR /scenic
RUN git checkout 01b082706e7304e699baaa1bf4d15dcbf4d52de2
RUN python3 -m pip install -vq .
WORKDIR /
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt

RUN mkdir -p variables
COPY variables_base_32.npy /variables/
COPY variables_base_16.npy /variables/
COPY variables_large_14.npy /variables/

COPY download.py /scenic/scenic/projects/baselines/clip/