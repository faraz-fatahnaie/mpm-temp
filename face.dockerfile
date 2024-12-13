FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list

# Install dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-venv python3.10-distutils ca-certificates g++ gcc make git libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

WORKDIR /app

RUN rm -rf /tmp/*

RUN pip install cmake
RUN pip install opencv-python==4.5.5.64
#RUN pip install mediapipe==0.9.1.0
RUN pip install scikit-image==0.19.3
RUN pip install onnxruntime-gpu==1.16.0

#RUN pip install confluent-kafka~=2.2.0
RUN pip install numpy==1.25.0
#RUN pip install Pillow~=10.0.0
#RUN pip install PyYAML
RUN pip install protobuf==3.20
#RUN pip install moviepy
RUN pip install sentence-transformers
RUN pip install librosa
RUN pip install soundfile
#RUN pip install Levenshtein
#RUN pip install transformers

# Install packages from ocr.txt
#COPY ./requirements/ocr.txt /app/requirements/ocr.txt
#RUN pip install -r /app/requirements/ocr.txt

FROM base

WORKDIR /app

COPY ./core ./core
COPY ./modules ./modules
COPY ./liveness_verify.py .

CMD ["python3", "liveness_verify.py"]
