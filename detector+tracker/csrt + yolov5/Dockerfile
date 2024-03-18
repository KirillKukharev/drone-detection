#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/
RUN apt-get update && apt-get install -y \
	build-essential \
	python3.8 \
	python3.8-dev\
	python3-pip \
	&& ln -s /usr/bin/pip3 /usr/bin/pip \
    	&& ln -s /usr/bin/python3.8 /usr/bin/python \
    	&& apt install --no-install-recommends -y gcc git zip curl libsm6 libxext6 htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 \
    	&& apt-get install -y libxrender-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com
    
RUN apt upgrade --no-install-recommends -y openssl tar
ENV PROJECT_PATH=/ultralytics/v1
WORKDIR $PROJECT_PATH

RUN git clone https://github.com/ultralytics/ultralytics.git -b main /usr/src/ultralytics/

RUN python -m pip install --upgrade pip wheel
RUN pip --version
RUN python --version

RUN pip install --no-cache paddlepaddle==2.4.2 x2paddle
RUN pip install --no-cache nvidia-pyindex
RUN pip install --no-cache nvidia-tensorrt
RUN pip install --no-cache onnx==1.15.0
RUN pip install --no-cache Cython==3.0.5
RUN pip install --no-cache Flask==3.0.0
RUN pip install --no-cache imutils==0.5.4
RUN pip install --no-cache matplotlib==3.7.3
RUN pip install --no-cache opencv-python==4.8.0.76
RUN pip install --no-cache opencv-contrib-python==4.2.0.34
RUN pip install --no-cache pandas==2.0.3
RUN pip install --no-cache pyyaml==6.0.1
RUN pip install --no-cache seaborn==0.13.0
RUN pip install --no-cache tqdm==4.66.1
RUN pip install --no-cache ultralytics==8.0.212
RUN pip install --no-cache numpy==1.24.4

RUN rm -rf tmp 
ENV OMP_NUM_THREADS=1
COPY . /ultralytics/v1
CMD ["python", "./run.py"]

