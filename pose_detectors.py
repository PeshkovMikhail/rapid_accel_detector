import ultralytics
import ViTPoseTrack.VitPose as vit
import numpy as np

def xyxy2xywh(boxes):
    return np.concatenate([(boxes[...,:2]+boxes[...,2:])/2,boxes[...,2:]-boxes[...,:2]],len(boxes.shape)-1)

class YOLODetector:
    def __init__(self):
        self.model = ultralytics.YOLO("yolov8l-pose")
    
    def track(self, image):
        results = self.model.track(image, persist = True, verbose = False, tracker="bytetrack.yaml")
        if results[0].boxes.id is None:
            return []
        boxes = results[0].boxes.xywh.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints = results[0].keypoints.xy.cpu().tolist()
        cls = results[0].boxes.cls.int().cpu().tolist()
        confs = results[0].keypoints.conf.cpu().tolist()

        res = list(map(lambda t: (t[0], t[1], t[2], t[4]), filter(lambda t: t[3] == 0, zip(boxes, track_ids, keypoints, cls, confs))))
        if len(res) == 0 or res is None:
            return []
        return res
    
class VitPoseDetector:
    def __init__(self) -> None:
        self.detect = vit.VitPose('YOLOX-s','ViTPose-B (multi-task train, COCO)',0.5,0.5)
        

    def track(self, image):
        res = self.detect.process_one(image)
        return map(lambda t: (xyxy2xywh(t['bbox'][:4]),t['track_id'],t['keypoints'][:,:2],t['keypoints'][:,2]),res)
    

        