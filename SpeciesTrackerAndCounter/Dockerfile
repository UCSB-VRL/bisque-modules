# FROM python:3.7.16-bullseye
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt install python3.7
RUN apt-get update
RUN apt-get install python3 python3-pip -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

# ===================Module Dependencies============================

RUN pip3 install cycler imageio kiwisolver matplotlib numpy Pillow pyparsing python-dateutil scipy rich

# ===================Copy Source Code===============================

RUN mkdir /module
WORKDIR /module
COPY requirements.txt /module

# ===============bqapi for python3 Dependencies=====================
# pip install in this exact order
RUN pip3 install six
RUN pip3 install lxml
RUN pip3 install cython
RUN pip3 install -r requirements.txt
RUN pip3 install requests==2.18.4
RUN pip3 install requests-toolbelt

# =====================Build Directory Structure====================

COPY src /module/src
COPY PythonScriptWrapper.py /module/
COPY bqapi/ /module/bqapi

# Replace the following line with your {ModuleName}.xml
COPY SpeciesTrackerAndCounter.xml /module/SpeciesTrackerAndCounter.xml

ENV PATH /module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH $PYTHONPATH:/module/src

