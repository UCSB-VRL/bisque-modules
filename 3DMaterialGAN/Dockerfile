FROM ubuntu:bionic
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update -y
RUN apt-get install -y build-essential
RUN apt-get install -y python3.6 python3.6-dev python3-pip && \
        apt-get install -y git  && \
        # update pip
        python3.6 -m pip install pip --upgrade && \
        python3.6 -m pip install wheel

RUN apt-get -y install python-pip liblapack3 libblas-dev liblapack-dev gfortran
RUN apt-get update
RUN pip install torch==1.2.0
RUN pip install numpy==1.14.5
RUN pip install tensorflow-gpu==1.13.1
RUN pip install tensorboardX==2.0
RUN pip install QtPy==1.9.0
RUN pip install lxml
#RUN pip install --user --install-option="--prefix=" -U scikit-learn==0.19.1
RUN pip install -i https://biodev.ece.ucsb.edu/py/bisque/prod/+simple bisque-api==0.5.9

WORKDIR /module
COPY src  /module/src

# ===============bqapi for python3 Dependencies=====================
# pip install in this exact order
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
COPY 3DMaterialGAN.xml /module/3DMaterialGAN.xml

ENV PATH /module:$PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV PYTHONPATH $PYTHONPATH:/module/src
