import onnxruntime
import copy

from pose_detectors import YOLODetector, VitPoseDetector
from preprocess import *
from config import *

KP_COUNT = 17

transforms = [
    UniformSampleFrames(POSEC3D_INPUT_FRAMES_COUNT),
    PoseDecode(),
    PoseCompact(hw_ratio=1., allow_imgpad=True),
    Resize(scale=(-1, 64)),
    CenterCrop(crop_size=64),
    GeneratePoseTarget(sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    FormatShape(input_format='NCTHW_Heatmap'),
    PackActionInputs()
]

class ActionDetector:
    def __init__(self, height, width, model_path):
        f = open(".pose_detector")
        pd = f.read()
        f.close()
        self.pose_detector = YOLODetector() if pd == "yolo" else VitPoseDetector()
        self.session = onnxruntime.InferenceSession(model_path)

        self.height = height
        self.width = width

        self.info_per_frame = {}


    def update_poses(self, image, frame_id):
        current_poses = {}
        
        for bbox, track_id, kp, conf in self.pose_detector.track(image):
            track_info= self.info_per_frame.get(track_id,
                {
                    "bboxes": np.zeros((POSEC3D_INPUT_FRAMES_COUNT, 4)),
                    "keypoints": np.zeros((1, POSEC3D_INPUT_FRAMES_COUNT, KP_COUNT, 2)),
                    "score": np.zeros((1, POSEC3D_INPUT_FRAMES_COUNT, KP_COUNT))
                })
            track_info['bboxes'][frame_id % POSEC3D_INPUT_FRAMES_COUNT] = np.array(bbox)
            current_poses[track_id] = np.zeros((KP_COUNT, 2))
            for i in range(KP_COUNT):
                track_info['keypoints'][0, frame_id % POSEC3D_INPUT_FRAMES_COUNT, i] = kp[i]
                track_info['score'][0, frame_id % POSEC3D_INPUT_FRAMES_COUNT, i] = conf[i]
                current_poses[track_id][i] = kp[i]
            self.info_per_frame[track_id] = track_info
        return current_poses
    

    def _get_label(self, seq):
        temp = copy.deepcopy(seq)
        for transform in transforms:
            temp = transform.transform(temp)
        input_feed = {'input_tensor': temp["inputs"].cpu().data.numpy()}
        outputs = self.session.run(['cls_score'], input_feed=input_feed)
        return np.argmax(torch.nn.functional.softmax(torch.tensor(outputs[0][0])).numpy())
    
    def classify(self):
        det_res = {}
        for k, v in self.info_per_frame.items():
            det_res[k] = (self._get_label({
                "frame_dir": "output000",
                "total_frames": POSEC3D_INPUT_FRAMES_COUNT,
                "img_shape": (self.height, self.width),
                "original_shape": (self.height, self.width),
                "label": 0,
                "keypoint": v['keypoints'],
                "keypoint_score": v['score']
            }), v['bboxes'])
        self.info_per_frame = {}
        return det_res