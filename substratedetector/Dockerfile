FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install pandas av opencv-python scikit-video
RUN pip install six lxml requests==2.18.4 requests-toolbelt

RUN mkdir /module
RUN mkdir /module/src

WORKDIR /module
COPY src /module/src
COPY PythonScriptWrapper.py /module
COPY bqapi /module/bqapi
COPY substratedetector.xml /module

ENV PATH /module:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/conda/bin
