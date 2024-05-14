ARG PYTORCH="2.2.2"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN pip3 install mmcv-full==1.7.2 mmengine==0.10.4 mmdet==2.26.0
ADD https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth /root/.cache/torch/hub/checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth
ADD https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth /root/.cache/torch/hub/checkpoints/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth
ADD https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth /root/.cache/torch/hub/checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
ADD https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth /root/.cache/torch/hub/checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth
ADD https://huggingface.co/public-data/ViTPose/resolve/main/models/vitpose-b-multi-coco.pth?download=true /root/.cache/huggingface/hub/models--public-data--ViTPose/snapshots/f29fe162c2b47eaaeb752b0665076026693e9ab4/models/vitpose-b-multi-coco.pth

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