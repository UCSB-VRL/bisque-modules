# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# ==================================================================

FROM python:3.6.15-buster

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# ===================Module Dependencies============================
RUN pip3 install cycler==0.10.0
RUN pip3 install dataclasses==0.8
RUN pip3 install decorator==4.4.2
RUN pip3 install future==0.18.2
RUN pip3 install imageio==2.9.0
RUN pip3 install kiwisolver==1.3.1
RUN pip3 install matplotlib==3.3.4
RUN pip3 install networkx==2.5
RUN pip3 install numpy==1.19.5
RUN pip3 install Pillow==8.1.2
#RUN pip3 install pkg-resources==0.0.0
RUN pip3 install pyparsing==2.4.7
RUN pip3 install python-dateutil==2.8.1
RUN pip3 install PyWavelets==1.1.1
RUN pip3 install scikit-image==0.17.2
RUN pip3 install scipy==1.5.4
RUN pip3 install six==1.15.0
RUN pip3 install tifffile==2020.9.3
RUN pip3 install torch==1.7.0
RUN pip3 install torchvision==0.8.0
RUN pip3 install tqdm==4.59.0
RUN pip3 install typing-extensions==3.7.4.3
RUN pip3 install PyYAML
RUN pip3 install h5py
RUN pip3 install Pillow

RUN pip install lxml
RUN pip install -i https://biodev.ece.ucsb.edu/py/bisque/prod/+simple bisque-api==0.5.9


RUN mkdir /module
WORKDIR /module

COPY src /module/src

# ===============bqapi for python3 Dependencies=====================
#pip install in this exact order
RUN pip install six
RUN pip install pyyaml scipy tiffile
RUN pip install requests==2.18.4
RUN pip install requests-toolbelt
# =====================Build Directory Structure====================
RUN alias python='python3.6'
COPY PythonScriptWrapper.py /module/
COPY bqapi/ /module/bqapi
RUN apt-get install zip -y
# Replace the following line with your {ModuleName}.xml
COPY EBSDSR.xml /module/EBSDSR.xml

ENV PATH /module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH $PYTHONPATH:/module/src

