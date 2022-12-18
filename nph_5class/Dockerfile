# FROM python:3.6.15-buster
# FROM nvidia/cuda:11.7.1-base-ubuntu20.04 
FROM python:3.8.13-buster

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get -y update   && \
    apt-get -y upgrade  && \
    apt-get -y install python3
RUN apt-get -y install python3-pip

RUN apt install -y vim
RUN apt install -y wget

# ===================Module Dependencies============================

RUN pip install torch torchvision numpy nibabel matplotlib scipy scikit-image SimpleITK
# RUN wget -q http://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
#     chmod 775 fslinstaller.py && \
#     python3 /fslinstaller.py -d /usr/local/fsl -V 6.0.4 -q && \
#     rm -f /fslinstaller.py

# RUN mv /usr/local/fsl /usr/local/bin/fsl


# ===================Copy Source Code===============================

RUN mkdir /module
WORKDIR /module

COPY src /module/src
# RUN chmod +x /module/src/skull_strip.sh

# ===============bqapi for python3 Dependencies=====================
# pip install in this exact order
RUN pip3 install six
RUN pip3 install lxml
RUN pip3 install requests==2.18.4
RUN pip3 install requests-toolbelt
RUN pip3 install tables

# =====================Build Directory Structure====================

EXPOSE 8080
EXPOSE 5000
COPY PythonScriptWrapper.py /module/
COPY bqapi/ /module/bqapi

# Replace the following line with your {ModuleName}.xml
COPY NPHSegmentation.xml /module/NPHSegmentation.xml

ENV PATH /module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#:/usr/local/fsl/bin
ENV PYTHONPATH $PYTHONPATH:/module/src
# ENV FSLOUTPUTTYPE NIFTI_GZ
