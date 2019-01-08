FROM xreki/paddle:cuda92_cudnn7
MAINTAINER Xreki <liuyiqun01@baidu.com>

COPY ./.vimrc /root/
COPY ./enable_core.sh /root/
RUN /bin/bash /root/enable_core.sh

RUN mkdir -p /opt && \
    cd /opt && \
    wget https://github.com/gperftools/gperftools/releases/download/gperftools-2.7/gperftools-2.7.tar.gz && \
    tar xvf gperftools-2.7.tar.gz && \
    cd /opt/gperftools-2.7 && \
    ./configure && \
    make -j12 && \
    make install

RUN git config --global http.sslverify false 
RUN go get -u github.com/google/pprof && \
ENV PATH=${GOPATH}/bin:$PATH

RUN apt-get update && \
    apt-get install -y vim gdb
