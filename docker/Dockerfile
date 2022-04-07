ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install nano

RUN conda clean --all
RUN pip install mmcv-full===1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html

RUN pip install mmdet==2.14.0

RUN git clone https://github.com/anhtv26062000/vietnamese_namecard_extraction.git /vietnamese_namecard_extraction
WORKDIR /vietnamese_namecard_extraction/libs/mmocr
ENV FORCE_CUDA="1"
RUN pip uninstall -y opencv-python && pip install -r requirements.txt
# RUN pip install --no-cache-dir -e .
WORKDIR /vietnamese_namecard_extraction
RUN pip install -r requirements.txt
