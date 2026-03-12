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

RUN pip3 install supervisely==6.73.486

RUN mkdir -p variables
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_base_32.npy /variables/variables_base_32.npy
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_base_16.npy /variables/variables_base_16.npy
COPY --from=supervisely/owl-vit:1.0.1-ruslan /variables/variables_large_14.npy /variables/variables_large_14.npy

RUN apt-get update && apt-get -y install curl

RUN curl -L -o /variables/bpe_simple_vocab_16e6.txt.gz "https://github.com/supervisely-ecosystem/serve-owl-vit/releases/download/v1.0.0/bpe_simple_vocab_16e6.txt.gz"

RUN git clone https://github.com/google-research/scenic.git
COPY download.py /scenic/scenic/projects/baselines/clip/
COPY tokenizer.py /scenic/scenic/projects/owl_vit/clip/

ENV PYTHONPATH=/scenic

LABEL python_sdk_version="6.73.486"
