import numpy as np
from config import *

NOSE:           int = 0
LEFT_EYE:       int = 1
RIGHT_EYE:      int = 2
LEFT_EAR:       int = 3
RIGHT_EAR:      int = 4
LEFT_SHOULDER:  int = 5
RIGHT_SHOULDER: int = 6
LEFT_ELBOW:     int = 7
RIGHT_ELBOW:    int = 8
LEFT_WRIST:     int = 9
RIGHT_WRIST:    int = 10
LEFT_HIP:       int = 11
RIGHT_HIP:      int = 12
LEFT_KNEE:      int = 13
RIGHT_KNEE:     int = 14
LEFT_ANKLE:     int = 15
RIGHT_ANKLE:    int = 16

HEAD_POINTS = [NOSE, RIGHT_EAR, RIGHT_EYE, LEFT_EAR, LEFT_EYE]
CHEST_POINTS = [RIGHT_SHOULDER, LEFT_SHOULDER, RIGHT_HIP, LEFT_HIP]

TRACK_POINTS = HEAD_POINTS if track_point == "head" else CHEST_POINTS

class TrackData:
    def __init__(self, frame_id, average_height_sectors, sector_len, period):
        self.height = 0
        self.height_count = 0
        self.speed_data = np.zeros((SPEED_WINDOW, 2))
        self.average_height_sectors = average_height_sectors
        self.period = period
        self.cursor = 0
        self.frame_id = frame_id
        self.prev_coords = None
        self.sector_len = sector_len

    def update_speed(self, kp, frame_id):
        coords = get_coords(kp)
        pixel_height = get_pixel_height(kp)

        if np.any(np.isnan(coords)) or np.isnan(pixel_height):
            return np.nan
        
        #update sector's average pixel height
        self.average_height_sectors[coords[1] // self.sector_len[1], coords[0] // self.sector_len[0]] += np.array([pixel_height, 1])

        height_sum, count = self.average_height_sectors[coords[1] // self.sector_len[1], coords[0] // self.sector_len[0]]
        average_pixel_height = height_sum / count
        height = pixel_height / average_pixel_height * AVERAGE_HEIGHT

        pixel_per_meter = pixel_height / height
        if self.prev_coords is None:
            self.prev_coords = coords
            return np.nan

        self.speed_data[self.cursor % SPEED_WINDOW] = (coords - self.prev_coords) / pixel_per_meter                                                             
        self.cursor += 1
        self.frame_id = frame_id
        self.prev_coords = coords

        return self.normal_speed()
    
    def vector_speed(self):
        speeds = self.speed_data[np.all(self.speed_data[:] != [0, 0], axis=1)]
        return np.sum(speeds, axis=0) / (self.period * speeds.shape[0])
    
    def normal_speed(self):
        speeds = np.abs(self.speed_data[np.all(self.speed_data[:] != [0, 0], axis=1)])
        return np.linalg.norm(np.sum(speeds, axis=0) / (self.period * speeds.shape[0] * FPS_DIVIDER))

def get_pixel_height(kp):

    head = kp[HEAD_POINTS]
    top = head[np.all(head[:] != [0, 0], axis=1)].transpose((1, 0)).mean(axis=1)[1]

    # legs' pixel length  
    right = kp[RIGHT_HIP][1] + np.linalg.norm(kp[RIGHT_HIP] - kp[RIGHT_KNEE]) + np.linalg.norm(kp[RIGHT_KNEE] - kp[RIGHT_ANKLE])
    left = kp[LEFT_HIP][1] + np.linalg.norm(kp[LEFT_HIP] - kp[LEFT_KNEE]) + np.linalg.norm(kp[LEFT_KNEE] - kp[LEFT_ANKLE])

    # check if ankle and hip points exist
    right_state = not np.all(kp[RIGHT_HIP] == [0, 0]) and not np.all(kp[RIGHT_ANKLE] == [0, 0]) and not np.all(kp[RIGHT_KNEE] == [0, 0])
    left_state = not np.all(kp[LEFT_HIP] == [0, 0]) and not np.all(kp[LEFT_ANKLE] == [0, 0]) and not np.all(kp[LEFT_KNEE] == [0, 0])
    if right_state and left_state:
        return top - max(right, left)
    elif right_state:
        return top - right
    elif left_state:
        return top - left
    return np.nan


def get_coords(kp):
    body = kp[TRACK_POINTS]
    return body[np.all(body[:] != [0, 0], axis=1)].transpose((1, 0)).mean(axis=1).astype(np.int32)



class SpeedTracker:
    def __init__(self, height, width, period) -> None:
        self.sector_len = np.array([width // WIDTH_SECTORS, height // HEIGHT_SECTORS])
        self.period = period
        self.average_speed = {}
        self.average_height_sectors = np.zeros((HEIGHT_SECTORS, WIDTH_SECTORS, 2))
    
    def speed(self, poses: dict, frame_id):
        res = {}
        for track_id, kp in poses.items():
            if track_id not in self.average_speed.keys():
                self.average_speed[track_id] = TrackData(frame_id, self.average_height_sectors, self.sector_len, self.period)
            if frame_id % FPS_DIVIDER == 0:
                res[track_id] = self.average_speed[track_id].update_speed(kp, frame_id)
                continue
            res[track_id] = self.average_speed[track_id].normal_speed()
        return res