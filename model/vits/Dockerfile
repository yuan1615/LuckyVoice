FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
MAINTAINER xin <xin.yuan@vxichina.com>
RUN apt-get update -y
RUN apt-get install -y libsndfile1-dev
COPY ./packages/ttsfrd-0.0.4-cp38-cp38-linux_x86_64.whl /tts/ttsfrd-0.0.4-cp38-cp38-linux_x86_64.whl
RUN pip install /tts/ttsfrd-0.0.4-cp38-cp38-linux_x86_64.whl
COPY . /tts
WORKDIR /tts
EXPOSE 7860
CMD chmod +777 /tts/run.sh &&\
/tts/run.sh