
# ===================Begin of the original Dockerfile===================

# FROM supervisely/base-py-sdk:6.70.22
# ENV DEBIAN_FRONTEND=noninteractive
# ENV XLA_PYTHON_CLIENT_PREALLOCATE=false

# RUN apt-get update
# RUN apt-get install -y git

# RUN pip3 install --upgrade pip
# RUN git clone https://github.com/google-research/scenic.git
# WORKDIR /scenic
# RUN git checkout 01b082706e7304e699baaa1bf4d15dcbf4d52de2
# RUN python3 -m pip install git+https://github.com/google-deepmind/optax.git@main
# RUN python3 -m pip install -vq .
# WORKDIR /
# COPY requirements.txt /tmp/requirements.txt
# RUN python3 -m pip install -r /tmp/requirements.txt

# RUN mkdir -p variables
# COPY --from=supervisely/owl-vit:1.0.1 /variables/variables_base_32.npy /variables/variables_base_32.npy
# COPY --from=supervisely/owl-vit:1.0.1 /variables/variables_base_16.npy /variables/variables_base_16.npy
# COPY --from=supervisely/owl-vit:1.0.1 /variables/variables_large_14.npy /variables/variables_large_14.npy

# COPY download.py /scenic/scenic/projects/baselines/clip/

# ===================End of the original Dockerfile===================


FROM supervisely/owl-vit:1.0.1-ruslan

RUN pip install --upgrade setuptools==69.5.1

# For CLIP:
RUN pip3 install torch==1.10.2
RUN pip3 install git+https://github.com/openai/CLIP.git
RUN pip3 install ml_collections==0.1.1
RUN pip3 install clu==0.0.9
RUN pip3 install tensorflow==2.12.0
RUN pip3 install tensorflow-datasets==4.9.2
RUN pip3 install scikit-image==0.17.1
RUN pip3 install bokeh==3.1.1
RUN pip3 install matplotlib==3.3.

# To import Sinkhorn matcher:
RUN pip3 install jaxlib==0.4.11
RUN pip3 install jax==0.4.13
RUN pip3 install ott-jax==0.2.0

RUN pip3 install supervisely==6.73.139

RUN mkdir -p variables
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_base_32.npy /variables/variables_base_32.npy
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_base_16.npy /variables/variables_base_16.npy
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_large_14.npy /variables/variables_large_14.npy

COPY download.py /scenic/scenic/projects/baselines/clip/