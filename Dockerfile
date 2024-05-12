ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
WORKDIR /vit
COPY . . 

# #VitPose
RUN pip3 install -e ViTPoseTrack
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 uninstall opencv-python-headless -y
RUN pip3 install opencv-python-headless

EXPOSE 3000

CMD ["python", "inference.py"]