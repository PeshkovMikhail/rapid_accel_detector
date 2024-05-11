from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
import torch.nn as nn
import huggingface_hub

from mmdet.apis import inference_detector, init_detector
from .mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker,STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
import numpy as np 


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def tracks2boxes(tracks: list[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections,
    tracks: list[STrack]
):
    if len(tracks) == 0:
        return []
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

class DetModel:
    MODEL_DICT = {
        'YOLOX-tiny': {
            'config':
            'ViTPoseTrack/mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
        },
        'YOLOX-s': {
            'config':
            'ViTPoseTrack/mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
        },
        'YOLOX-l': {
            'config':
            'ViTPoseTrack/mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
        },
        'YOLOX-x': {
            'config':
            'ViTPoseTrack/mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
            'model':
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
        },
    }

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self._load_all_models_once()
        self.model_name = 'YOLOX-l'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        d = self.MODEL_DICT[name]
        return init_detector(d['config'], d['model'], device=self.device)

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        out = inference_detector(self.model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)] * 79

        vis = self.model.show_result(image,
                                     person_det,
                                     score_thr=score_threshold,
                                     bbox_color=None,
                                     text_color=(200, 200, 200),
                                     mask_color=None)
        return vis


class PoseModel:
    MODEL_DICT = {
        'ViTPose-S (multi-task train, COCO)': {
            'config':
            'ViTPoseTrack/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_small_coco_256x192.py',
            'model': 'models/vitpose-b.pth',
        },
        'ViTPose-B (single-task train)': {
            'config':
            'ViTPoseTrack/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b.pth',
        },
        'ViTPose-L (single-task train)': {
            'config':
            'ViTPoseTrack/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l.pth',
        },
        'ViTPose-B (multi-task train, COCO)': {
            'config':
            'ViTPoseTrack/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            'model': 'models/vitpose-b-multi-coco.pth',
        },
        'ViTPose-L (multi-task train, COCO)': {
            'config':
            'ViTPoseTrack/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
            'model': 'models/vitpose-l-multi-coco.pth',
        },
    }

    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'ViTPose-B (multi-task train, COCO)'
        self.model = self._load_model(self.model_name)

    def _load_all_models_once(self) -> None:
        for name in self.MODEL_DICT:
            self._load_model(name)

    def _load_model(self, name: str) -> nn.Module:
        d = self.MODEL_DICT[name]
        ckpt_path = huggingface_hub.hf_hub_download('public-data/ViTPose',
                                                    d['model'])
        model = init_pose_model(d['config'], ckpt_path,device=self.device)
        return model

    def set_model(self, name: str) -> None:
        if name == self.model_name:
            return
        self.model_name = name
        self.model = self._load_model(name)

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
    
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(self.model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[dict[str, np.ndarray]],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:

        vis = vis_pose_result(self.model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis


class VitPose:
    def __init__(self, det_model_name: str, pose_model_name: str,
        box_score_threshold: float,
        kpt_score_threshold: float, 
        visualize : bool = False,
        vis_dot_radius: int = 0,
        vis_line_thickness: int = 0):
        self.det_model = DetModel()
        self.pose_model = PoseModel()
        self.det_model.set_model(det_model_name)
        self.pose_model.set_model(pose_model_name)

        self.track = BYTETracker(BYTETrackerArgs())
        
        self.box_score_threshold = box_score_threshold
        self.kpt_score_threshold = kpt_score_threshold

        self.visualize = visualize
        self.vis_line_thickness = vis_line_thickness
        self.vis_dot_radius = vis_dot_radius

    def tracker_clear(self):
        del self.track
        self.track = BYTETracker(BYTETrackerArgs())

    def process_vid(
        self, video_path: str
    ) -> tuple[str, list[list[dict[str, np.ndarray]]]]:
        if video_path is None:
            return

        cap = cv2.VideoCapture(video_path)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        preds_all = []

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        writer = cv2.VideoWriter(out_file.name, fourcc, fps, (width, height))


        for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            ok, frame = cap.read()
            print("current_frame: ",i)
            if not ok:
                break
            det_preds = self.det_model.detect(frame)
            shape = frame.shape[:2]
            
            if self.visualize:
                preds, vis = self.pose_model.predict_pose_and_visualize(
                    frame, det_preds, self.box_score_threshold, self.kpt_score_threshold,
                    self.vis_dot_radius, self.vis_line_thickness)
            else:
                preds, vis = self.pose_model.predict_pose(frame, det_preds, self.box_score_threshold)
            
            det_preds = det_preds[0][det_preds[0][:,4]>self.box_score_threshold]

            track_ids = match_detections_with_tracks(det_preds[:,:4],self.track.update(det_preds,shape,shape))
            for j in range(len(preds)):                
                preds[j]["track_id"] = track_ids[j]

            preds_all.append(preds)
            if self.visualize:
                writer.write(vis)          
        cap.release()
        writer.release()

        self.tracker_clear()
        
        if self.visualize:
            return out_file.name, preds_all
        return preds_all
    
    def process_one(self,img):
        if self.visualize:
            det_preds = self.det_model.detect(img)
            preds, vis = self.pose_model.predict_pose_and_visualize(
                img, det_preds, self.box_score_threshold,  self.kpt_score_threshold,
                 self.vis_dot_radius,  self.vis_line_thickness)
            
            det_preds = det_preds[0][det_preds[0][:,4]>self.box_score_threshold]
            shape = img.shape[:2]
            track_ids = match_detections_with_tracks(det_preds[:,:4],self.track.update(det_preds,shape,shape))
            for j in range(len(preds)):                
                preds[j]["track_id"] = track_ids[j]
            
            return vis,preds
        
        det_preds = self.det_model.detect(img)
        preds = self.pose_model.predict_pose(
            img, det_preds, self.box_score_threshold)
        
        det_preds = det_preds[0][det_preds[0][:,4]>self.box_score_threshold]
        shape = img.shape[:2]
        track_ids = match_detections_with_tracks(det_preds[:,:4],self.track.update(det_preds,shape,shape))
        for j in range(len(preds)):                
            preds[j]["track_id"] = track_ids[j]
        
        return preds


    
