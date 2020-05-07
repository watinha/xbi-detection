FROM python:3.7.3-alpine
ADD . /app
WORKDIR /app
EXPOSE 3000
RUN apk add --update python3 python3-dev gfortran py-pip build-base g++ gfortran file binutils \
                     musl-dev openblas-dev libstdc++ openblas libpng-dev freetype-dev
RUN apk add fontconfig \
            py3-dateutil \
            py3-decorator \
            py3-defusedxml \
            py3-jinja2 \
            py3-jsonschema \
            py3-markupsafe \
            py3-pexpect \
            py3-prompt_toolkit \
            py3-pygments \
            py3-ptyprocess \
            py3-six \
            py3-tornado \
            py3-wcwidth \
            py3-zmq \
            tini
RUN apk add pkgconfig wget
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install np
RUN pip3 install 'scipy<1.4'
RUN pip3 install Cython
RUN pip3 install sklearn
RUN pip3 install nltk
RUN python3 setup.py
RUN pip3 install pandas
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install liac-arff
RUN pip3 install imblearn

CMD ["ash"]
