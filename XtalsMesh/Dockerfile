FROM jonathanhestroffer/xtalmesh
ARG DEBIAN_FRONTEND="noninteractive"


RUN mkdir /module
RUN mkdir /work
WORKDIR /work
COPY ./bqapi /work/bqapi
COPY ./public /work
COPY Xtal_Mesher.py .
COPY Xtal_Smoother.py .
COPY tet_mesh_l2q.cpp .
COPY tet_mesh_l2q .
COPY tet_mesh_l2q.sh .

COPY PythonScriptWrapper /work/
COPY PythonScriptWrapper.py /work/

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install requests-toolbelt requests==2.10.0

ENV PATH /work:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
CMD [ 'PythonScriptWrapper' ]
