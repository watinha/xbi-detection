FROM python:3.10.2-alpine3.15
RUN apk add python3-dev \
            g++ \
            jpeg-dev \
            openblas-dev \
            lapack-dev \
            build-base \
            tzdata


RUN python -m pip install --upgrade pip
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install np
RUN pip install scipy
RUN pip install Cython
RUN pip install scikit-learn
RUN pip install nltk
RUN pip install pandas

RUN apk add zlib-dev
RUN pip install seaborn
RUN pip install matplotlib
RUN pip install liac-arff
RUN pip install imbalanced-learn
RUN pip install Pillow
RUN cp /usr/share/zoneinfo/America/Sao_Paulo /etc/localtime
RUN echo "America/Sao_Paulo" > /etc/timezone

CMD ["ash"]
