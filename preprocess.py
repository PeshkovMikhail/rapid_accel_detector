import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple, Union
from torch.nn.modules.utils import _pair
import torch
from collections.abc import Sized
import cv2
from typing import Any
import itertools

from base_data_element import BaseDataElement

BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
IndexType: Union[Any] = Union[str, slice, int, list, LongTypeTensor,
                              BoolTypeTensor, np.ndarray]

def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | int | tuple(float) | tuple(int)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)

def rescale_size(old_size: tuple,
                 scale: Union[float, int, Tuple[int, int]],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | int | tuple[int]): The scaling factor or maximum size.
            If it is a float number or an integer, then the image will be
            rescaled by this factor, else if it is a tuple of 2 integers, then
            the image will be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def _init_lazy_if_proper(results, lazy):
    """Initialize lazy operation properly.

    Make sure that a lazy operation is properly initialized,
    and avoid a non-lazy operation accidentally getting mixed in.

    Required keys in results are "imgs" if "img_shape" not in results,
    otherwise, Required keys in results are "img_shape", add or modified keys
    are "img_shape", "lazy".
    Add or modified keys in "lazy" are "original_shape", "crop_bbox", "flip",
    "flip_direction", "interpolation".

    Args:
        results (dict): A dict stores data pipeline result.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    if 'img_shape' not in results:
        results['img_shape'] = results['imgs'][0].shape[:2]
    if lazy:
        if 'lazy' not in results:
            img_h, img_w = results['img_shape']
            lazyop = dict()
            lazyop['original_shape'] = results['img_shape']
            lazyop['crop_bbox'] = np.array([0, 0, img_w, img_h],
                                           dtype=np.float32)
            lazyop['flip'] = False
            lazyop['flip_direction'] = None
            lazyop['interpolation'] = None
            results['lazy'] = lazyop
    else:
        assert 'lazy' not in results, 'Use Fuse after lazy operations'

class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required Keys:

        - total_frames
        - start_index (optional)

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips
        - clip_len

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        seed (int): The random seed used during test time. Defaults to 255.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False,
                 seed: int = 255) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_test_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for testing clips.
        """

        np.random.seed(self.seed)
        all_inds = []
        for i in range(self.num_clips):
            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips \
                    else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`UniformSampleFrames`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int64)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str

class PoseDecode:
    """Load and decode pose with given indices.

    Required Keys:

        - keypoint
        - total_frames (optional)
        - frame_inds (optional)
        - offset (optional)
        - keypoint_score (optional)

    Modified Keys:

        - keypoint
        - keypoint_score (optional)
    """

    @staticmethod
    def _load_kp(kp: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoints according to sampled indexes."""
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore: np.ndarray,
                      frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoint scores according to sampled indexes."""
        return kpscore[:, frame_inds].astype(np.float32)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PoseDecode`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'total_frames' not in results:
            results['total_frames'] = results['keypoint'].shape[1]

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(
                results['keypoint_score'], frame_inds)

        results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


def _combine_quadruple(a, b):
    return a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3]

class PoseCompact:
    """Convert the coordinates of keypoints to make it more compact.
    Specifically, it first find a tight bounding box that surrounds all joints
    in each frame, then we expand the tight box by a given padding ratio. For
    example, if 'padding == 0.25', then the expanded box has unchanged center,
    and 1.25x width and height.

    Required Keys:

        - keypoint
        - img_shape

    Modified Keys:

        - img_shape
        - keypoint

    Added Keys:

        - crop_quadruple

    Args:
        padding (float): The padding size. Defaults to 0.25.
        threshold (int): The threshold for the tight bounding box. If the width
            or height of the tight bounding box is smaller than the threshold,
            we do not perform the compact operation. Defaults to 10.
        hw_ratio (float | tuple[float] | None): The hw_ratio of the expanded
            box. Float indicates the specific ratio and tuple indicates a
            ratio range. If set as None, it means there is no requirement on
            hw_ratio. Defaults to None.
        allow_imgpad (bool): Whether to allow expanding the box outside the
            image to meet the hw_ratio requirement. Defaults to True.
    """

    def __init__(self,
                 padding: float = 0.25,
                 threshold: int = 10,
                 hw_ratio: Optional[Union[float, Tuple[float]]] = None,
                 allow_imgpad: bool = True) -> None:

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def transform(self, results: Dict) -> Dict:
        """Convert the coordinates of keypoints to make it more compact.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


class Resize:
    """Resize images to a specific size.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "img_shape", "keep_ratio",
    "scale_factor", "lazy", "resize_size". Required keys in "lazy" is None,
    added or modified key is "interpolation".

    Args:
        scale (float | Tuple[int]): If keep_ratio is True, it serves as scaling
            factor or maximum size:
            If it is a float number, the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, the image will
            be rescaled as large as possible within the scale.
            Otherwise, it serves as (w, h) of output size.
        keep_ratio (bool): If set to True, Images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: True.
        interpolation (str): Algorithm used for interpolation,
            accepted values are "nearest", "bilinear", "bicubic", "area",
            "lanczos". Default: "bilinear".
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self,
                 scale,
                 keep_ratio=True,
                 interpolation='bilinear',
                 lazy=False):
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        else:
            raise TypeError(
                f'Scale must be float or tuple of int, but got {type(scale)}')
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.lazy = lazy

    def _resize_imgs(self, imgs, new_w, new_h):
        """Static method for resizing keypoint."""
        return [
            cv2.resize(img, (new_w, new_h))
            for img in imgs
        ]

    @staticmethod
    def _resize_kps(kps, scale_factor):
        """Static method for resizing keypoint."""
        return kps * scale_factor

    @staticmethod
    def _box_resize(box, scale_factor):
        """Rescale the bounding boxes according to the scale_factor.

        Args:
            box (np.ndarray): The bounding boxes.
            scale_factor (np.ndarray): The scale factor used for rescaling.
        """
        assert len(scale_factor) == 2
        scale_factor = np.concatenate([scale_factor, scale_factor])
        return box * scale_factor

    def transform(self, results):
        """Performs the Resize augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        if 'scale_factor' not in results:
            results['scale_factor'] = np.array([1, 1], dtype=np.float32)
        img_h, img_w = results['img_shape']

        if self.keep_ratio:
            new_w, new_h = rescale_size((img_w, img_h), self.scale)
        else:
            new_w, new_h = self.scale

        self.scale_factor = np.array([new_w / img_w, new_h / img_h],
                                     dtype=np.float32)

        results['img_shape'] = (new_h, new_w)
        results['keep_ratio'] = self.keep_ratio
        results['scale_factor'] = results['scale_factor'] * self.scale_factor

        if not self.lazy:
            if 'imgs' in results:
                results['imgs'] = self._resize_imgs(results['imgs'], new_w,
                                                    new_h)
            if 'keypoint' in results:
                results['keypoint'] = self._resize_kps(results['keypoint'],
                                                       self.scale_factor)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')
            lazyop['interpolation'] = self.interpolation

        if 'gt_bboxes' in results:
            assert not self.lazy
            results['gt_bboxes'] = self._box_resize(results['gt_bboxes'],
                                                    self.scale_factor)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_resize(
                    results['proposals'], self.scale_factor)

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'scale={self.scale}, keep_ratio={self.keep_ratio}, '
                    f'interpolation={self.interpolation}, '
                    f'lazy={self.lazy})')
        return repr_str

class RandomCrop:
    """Vanilla square random crop that specifics the output size.

    Required keys in results are "img_shape", "keypoint" (optional), "imgs"
    (optional), added or modified keys are "keypoint", "imgs", "lazy"; Required
    keys in "lazy" are "flip", "crop_bbox", added or modified key is
    "crop_bbox".

    Args:
        size (int): The output size of the images.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, size, lazy=False):
        if not isinstance(size, int):
            raise TypeError(f'Size must be an int, but got {type(size)}')
        self.size = size
        self.lazy = lazy

    @staticmethod
    def _crop_kps(kps, crop_bbox):
        """Static method for cropping keypoint."""
        return kps - crop_bbox[:2]

    @staticmethod
    def _crop_imgs(imgs, crop_bbox):
        """Static method for cropping images."""
        x1, y1, x2, y2 = crop_bbox
        return [img[y1:y2, x1:x2] for img in imgs]

    @staticmethod
    def _box_crop(box, crop_bbox):
        """Crop the bounding boxes according to the crop_bbox.

        Args:
            box (np.ndarray): The bounding boxes.
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """

        x1, y1, x2, y2 = crop_bbox
        img_w, img_h = x2 - x1, y2 - y1

        box_ = box.copy()
        box_[..., 0::2] = np.clip(box[..., 0::2] - x1, 0, img_w - 1)
        box_[..., 1::2] = np.clip(box[..., 1::2] - y1, 0, img_h - 1)
        return box_

    def _all_box_crop(self, results, crop_bbox):
        """Crop the gt_bboxes and proposals in results according to crop_bbox.

        Args:
            results (dict): All information about the sample, which contain
                'gt_bboxes' and 'proposals' (optional).
            crop_bbox(np.ndarray): The bbox used to crop the original image.
        """
        results['gt_bboxes'] = self._box_crop(results['gt_bboxes'], crop_bbox)
        if 'proposals' in results and results['proposals'] is not None:
            assert results['proposals'].shape[1] == 4
            results['proposals'] = self._box_crop(results['proposals'],
                                                  crop_bbox)
        return results

    def transform(self, results):
        """Performs the RandomCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        assert self.size <= img_h and self.size <= img_w

        y_offset = 0
        x_offset = 0
        if img_h > self.size:
            y_offset = int(np.random.randint(0, img_h - self.size))
        if img_w > self.size:
            x_offset = int(np.random.randint(0, img_w - self.size))

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = x_offset / img_w, y_offset / img_h
        w_ratio, h_ratio = self.size / img_w, self.size / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        new_h, new_w = self.size, self.size

        crop_bbox = np.array(
            [x_offset, y_offset, x_offset + new_w, y_offset + new_h])
        results['crop_bbox'] = crop_bbox

        results['img_shape'] = (new_h, new_w)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = x_offset * (lazy_right - lazy_left) / img_w
            right = (x_offset + new_w) * (lazy_right - lazy_left) / img_w
            top = y_offset * (lazy_bottom - lazy_top) / img_h
            bottom = (y_offset + new_h) * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        # Process entity boxes
        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(size={self.size}, '
                    f'lazy={self.lazy})')
        return repr_str

class CenterCrop(RandomCrop):
    """Crop the center area from images.

    Required keys are "img_shape", "imgs" (optional), "keypoint" (optional),
    added or modified keys are "imgs", "keypoint", "crop_bbox", "lazy" and
    "img_shape". Required keys in "lazy" is "crop_bbox", added or modified key
    is "crop_bbox".

    Args:
        crop_size (int | tuple[int]): (w, h) of crop size.
        lazy (bool): Determine whether to apply lazy operation. Default: False.
    """

    def __init__(self, crop_size, lazy=False):
        self.crop_size = _pair(crop_size)
        self.lazy = lazy
        # if not mmengine.is_tuple_of(self.crop_size, int):
        #     raise TypeError(f'Crop_size must be int or tuple of int, '
        #                     f'but got {type(crop_size)}')

    def transform(self, results):
        """Performs the CenterCrop augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        _init_lazy_if_proper(results, self.lazy)
        if 'keypoint' in results:
            assert not self.lazy, ('Keypoint Augmentations are not compatible '
                                   'with lazy == True')

        img_h, img_w = results['img_shape']
        crop_w, crop_h = self.crop_size

        left = (img_w - crop_w) // 2
        top = (img_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        results['crop_bbox'] = crop_bbox
        results['img_shape'] = (new_h, new_w)

        if 'crop_quadruple' not in results:
            results['crop_quadruple'] = np.array(
                [0, 0, 1, 1],  # x, y, w, h
                dtype=np.float32)

        x_ratio, y_ratio = left / img_w, top / img_h
        w_ratio, h_ratio = new_w / img_w, new_h / img_h

        old_crop_quadruple = results['crop_quadruple']
        old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
        old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
        new_crop_quadruple = [
            old_x_ratio + x_ratio * old_w_ratio,
            old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,
            h_ratio * old_h_ratio
        ]
        results['crop_quadruple'] = np.array(
            new_crop_quadruple, dtype=np.float32)

        if not self.lazy:
            if 'keypoint' in results:
                results['keypoint'] = self._crop_kps(results['keypoint'],
                                                     crop_bbox)
            if 'imgs' in results:
                results['imgs'] = self._crop_imgs(results['imgs'], crop_bbox)
        else:
            lazyop = results['lazy']
            if lazyop['flip']:
                raise NotImplementedError('Put Flip at last for now')

            # record crop_bbox in lazyop dict to ensure only crop once in Fuse
            lazy_left, lazy_top, lazy_right, lazy_bottom = lazyop['crop_bbox']
            left = left * (lazy_right - lazy_left) / img_w
            right = right * (lazy_right - lazy_left) / img_w
            top = top * (lazy_bottom - lazy_top) / img_h
            bottom = bottom * (lazy_bottom - lazy_top) / img_h
            lazyop['crop_bbox'] = np.array([(lazy_left + left),
                                            (lazy_top + top),
                                            (lazy_left + right),
                                            (lazy_top + bottom)],
                                           dtype=np.float32)

        if 'gt_bboxes' in results:
            assert not self.lazy
            results = self._all_box_crop(results, results['crop_bbox'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(crop_size={self.crop_size}, '
                    f'lazy={self.lazy})')
        return repr_str


class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required Keys:

        - keypoint
        - keypoint_score (optional)
        - img_shape

    Added Keys:

        - imgs (optional)
        - heatmap_imgs (optional)

    Args:
        sigma (float): The sigma of the generated gaussian map.
            Defaults to 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Defaults to True.
        with_kp (bool): Generate pseudo heatmaps for keypoints.
            Defaults to True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Defaults to False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Defaults to ``((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                         (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                         (13, 15), (6, 12), (12, 14), (14, 16), (11, 12))``,
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Defaults to False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Defaults to (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Defaults to (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
        left_limb (tuple[int]): Indexes of left limbs, which is used when
            flipping heatmaps. Defaults to (0, 2, 4, 5, 6, 10, 11, 12),
            which is left limbs of skeletons we defined for COCO-17p.
        right_limb (tuple[int]): Indexes of right limbs, which is used when
            flipping heatmaps. Defaults to (1, 3, 7, 8, 9, 13, 14, 15),
            which is right limbs of skeletons we defined for COCO-17p.
        scaling (float): The ratio to scale the heatmaps. Defaults to 1.
    """

    def __init__(self,
                 sigma: float = 0.6,
                 use_score: bool = True,
                 with_kp: bool = True,
                 with_limb: bool = False,
                 skeletons: Tuple[Tuple[int]] = ((0, 1), (0, 2), (1, 3),
                                                 (2, 4), (0, 5), (5, 7),
                                                 (7, 9), (0, 6), (6, 8),
                                                 (8, 10), (5, 11), (11, 13),
                                                 (13, 15), (6, 12), (12, 14),
                                                 (14, 16), (11, 12)),
                 double: bool = False,
                 left_kp: Tuple[int] = (1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp: Tuple[int] = (2, 4, 6, 8, 10, 12, 14, 16),
                 left_limb: Tuple[int] = (0, 2, 4, 5, 6, 10, 11, 12),
                 right_limb: Tuple[int] = (1, 3, 7, 8, 9, 13, 14, 15),
                 scaling: float = 1.) -> None:

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons
        self.left_limb = left_limb
        self.right_limb = right_limb
        self.scaling = scaling

    def generate_a_heatmap(self, arr: np.ndarray, centers: np.ndarray,
                           max_values: np.ndarray) -> None:
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: img_h * img_w.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons). Shape: M * 2.
            max_values (np.ndarray): The max values of each keypoint. Shape: M.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for center, max_value in zip(centers, max_values):
            if max_value < self.eps:
                continue

            mu_x, mu_y = center[0], center[1]
            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            arr[st_y:ed_y, st_x:ed_x] = \
                np.maximum(arr[st_y:ed_y, st_x:ed_x], patch)

    def generate_a_limb_heatmap(self, arr: np.ndarray, starts: np.ndarray,
                                ends: np.ndarray, start_values: np.ndarray,
                                end_values: np.ndarray) -> None:
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: img_h * img_w.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs. Shape: M * 2.
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs. Shape: M * 2.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs. Shape: M.
            end_values (np.ndarray): The max values of the other keypoint
                in the corresponding limbs. Shape: M.
        """

        sigma = self.sigma
        img_h, img_w = arr.shape

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                self.generate_a_heatmap(arr, start[None], start_value[None])
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            arr[min_y:max_y, min_x:max_x] = \
                np.maximum(arr[min_y:max_y, min_x:max_x], patch)

    def generate_heatmap(self, arr: np.ndarray, kps: np.ndarray,
                         max_values: np.ndarray) -> None:
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            arr (np.ndarray): The array to store the generated heatmaps.
                Shape: V * img_h * img_w.
            kps (np.ndarray): The coordinates of keypoints in this frame.
                Shape: M * V * 2.
            max_values (np.ndarray): The confidence score of each keypoint.
                Shape: M * V.
        """

        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                self.generate_a_heatmap(arr[i], kps[:, i], max_values[:, i])

        if self.with_limb:
            for i, limb in enumerate(self.skeletons):
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                self.generate_a_limb_heatmap(arr[i], starts, ends,
                                             start_values, end_values)

    def gen_an_aug(self, results: Dict) -> np.ndarray:
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            np.ndarray: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint'].astype(np.float32)
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']

        # scale img_h, img_w and kps
        img_h = int(img_h * self.scaling + 0.5)
        img_w = int(img_w * self.scaling + 0.5)
        all_kps[..., :2] *= self.scaling

        num_frame = kp_shape[1]
        num_c = 0
        if self.with_kp:
            num_c += all_kps.shape[2]
        if self.with_limb:
            num_c += len(self.skeletons)

        ret = np.zeros([num_frame, num_c, img_h, img_w], dtype=np.float32)

        for i in range(num_frame):
            # M, V, C
            kps = all_kps[:, i]
            # M, C
            kpscores = all_kpscores[:, i] if self.use_score else \
                np.ones_like(all_kpscores[:, i])

            self.generate_heatmap(ret[i], kps, kpscores)
        return ret

    def transform(self, results: Dict) -> Dict:
        """Generate pseudo heatmaps based on joint coordinates and confidence.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        heatmap = self.gen_an_aug(results)
        key = 'heatmap_imgs' if 'imgs' in results else 'imgs'

        if self.double:
            indices = np.arange(heatmap.shape[1], dtype=np.int64)
            left, right = (self.left_kp, self.right_kp) if self.with_kp else (
                self.left_limb, self.right_limb)
            for l, r in zip(left, right):  # noqa: E741
                indices[l] = r
                indices[r] = l
            heatmap_flip = heatmap[..., ::-1][:, indices]
            heatmap = np.concatenate([heatmap, heatmap_flip])
        results[key] = heatmap
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'sigma={self.sigma}, '
                    f'use_score={self.use_score}, '
                    f'with_kp={self.with_kp}, '
                    f'with_limb={self.with_limb}, '
                    f'skeletons={self.skeletons}, '
                    f'double={self.double}, '
                    f'left_kp={self.left_kp}, '
                    f'right_kp={self.right_kp}, '
                    f'left_limb={self.left_limb}, '
                    f'right_limb={self.right_limb}, '
                    f'scaling={self.scaling})')
        return repr_str
    
class FormatShape:
    """Format final imgs shape to the given input_format.

    Required keys:

        - imgs (optional)
        - heatmap_imgs (optional)
        - modality (optional)
        - num_clips
        - clip_len

    Modified Keys:

        - imgs

    Added Keys:

        - input_shape
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NCTHW', 'NCHW', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            if 'imgs' in results:
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            if 'modality' in results and results['modality'] == 'Flow':
                clip_len = results['clip_len']
                imgs = imgs.reshape((-1, clip_len * imgs.shape[1]) +
                                    imgs.shape[2:])
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]
SCORE_TYPE = Union[torch.Tensor, np.ndarray, Sequence, Dict]


def format_label(value: LABEL_TYPE) -> torch.Tensor:
    """Convert various python types to label-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | int): Label value.

    Returns:
        :obj:`torch.Tensor`: The formatted label tensor.
    """

    # Handle single number
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim == 0:
        value = int(value.item())

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).to(torch.long)
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).to(torch.long)
    elif isinstance(value, int):
        value = torch.LongTensor([value])
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')

    return value


def format_score(value: SCORE_TYPE) -> Union[torch.Tensor, Dict]:
    """Convert various python types to score-format tensor.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`.

    Args:
        value (torch.Tensor | numpy.ndarray | Sequence | dict):
            Score values or dict of scores values.

    Returns:
        :obj:`torch.Tensor` | dict: The formatted scores.
    """

    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value).float()
    elif isinstance(value, Sequence) and not isinstance(value, str):
        value = torch.tensor(value).float()
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = format_score(v)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'Type {type(value)} is not an available label type.')

    return value

class InstanceData:
    """Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    InstanceData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. The type of value
    in data field can be base data structure such as `torch.Tensor`, `numpy.ndarray`, `list`, `str`, `tuple`,
    and can be customized data structure that has ``__len__``, ``__getitem__`` and ``cat`` attributes.

    Examples:
        >>> # custom data structure
        >>> class TmpObject:
        ...     def __init__(self, tmp) -> None:
        ...         assert isinstance(tmp, list)
        ...         self.tmp = tmp
        ...     def __len__(self):
        ...         return len(self.tmp)
        ...     def __getitem__(self, item):
        ...         if isinstance(item, int):
        ...             if item >= len(self) or item < -len(self):  # type:ignore
        ...                 raise IndexError(f'Index {item} out of range!')
        ...             else:
        ...                 # keep the dimension
        ...                 item = slice(item, None, len(self))
        ...         return TmpObject(self.tmp[item])
        ...     @staticmethod
        ...     def cat(tmp_objs):
        ...         assert all(isinstance(results, TmpObject) for results in tmp_objs)
        ...         if len(tmp_objs) == 1:
        ...             return tmp_objs[0]
        ...         tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        ...         tmp_list = list(itertools.chain(*tmp_list))
        ...         new_data = TmpObject(tmp_list)
        ...         return new_data
        ...     def __repr__(self):
        ...         return str(self.tmp)
        >>> from mmengine.structures import InstanceData
        >>> import numpy as np
        >>> import torch
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = InstanceData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = torch.Tensor([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> len(instance_data)
        2
        >>> print(instance_data)
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2])
            det_scores: tensor([0.8000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            polygons: [[1, 2, 3, 4]]
        ) at 0x7f64ecf0ec40>
        >>> print(instance_data[instance_data.det_scores > 1])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([], dtype=torch.int64)
            det_scores: tensor([])
            bboxes: tensor([], size=(0, 4))
            polygons: []
        ) at 0x7f660a6a7f70>
        >>> print(instance_data.cat([instance_data, instance_data]))
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3, 2, 3])
            det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263],
                        [0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7f203542feb0>
    """

    def __setattr__(self, name: str, value: Sized):
        """setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of ' \
                                                f'values {len(value)} is ' \
                                                'not consistent with ' \
                                                'the length of this ' \
                                                ':obj:`InstanceData` ' \
                                                f'{len(self)}'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> 'InstanceData': # type: ignore
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        assert isinstance(item, IndexType.__args__)
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `torch.Tensor` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, torch.Tensor):
            assert item.dim() == 1, 'Only support to get the' \
                                    ' values along the first dimension.'
            if isinstance(item, BoolTypeTensor.__args__):
                assert len(item) == len(self), 'The shape of the ' \
                                               'input(BoolTensor) ' \
                                               f'{len(item)} ' \
                                               'does not match the shape ' \
                                               'of the indexed tensor ' \
                                               'in results_field ' \
                                               f'{len(self)} at ' \
                                               'first dimension.'

            for k, v in self.items():
                if isinstance(v, torch.Tensor):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            :obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list}) \
               == 1 and len(set(itertools.chain(*field_keys_list))) \
               == len(field_keys_list[0]), 'There are different keys in ' \
                                           '`instances_list`, which may ' \
                                           'cause the cat operation ' \
                                           'to fail. Please make sure all ' \
                                           'elements in `instances_list` ' \
                                           'have the exact same key.'

        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, 'cat'):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0
        

class ActionDataSample(BaseDataElement):

    def set_gt_label(self, value: LABEL_TYPE) -> 'ActionDataSample':
        """Set `gt_label``."""
        self.set_field(format_label(value), 'gt_label', dtype=torch.Tensor)
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> 'ActionDataSample':
        """Set ``pred_label``."""
        self.set_field(format_label(value), 'pred_label', dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE) -> 'ActionDataSample':
        """Set score of ``pred_label``."""
        score = format_score(value)
        self.set_field(score, 'pred_score')
        if hasattr(self, 'num_classes'):
            assert len(score) == self.num_classes, \
                f'The length of score {len(score)} should be '\
                f'equal to the num_classes {self.num_classes}.'
        else:
            self.set_field(
                name='num_classes', value=len(score), field_type='metainfo')
        return self

    @property
    def proposals(self):
        """Property of `proposals`"""
        return self._proposals

    @proposals.setter
    def proposals(self, value):
        """Setter of `proposals`"""
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        """Deleter of `proposals`"""
        del self._proposals

    @property
    def gt_instances(self):
        """Property of `gt_instances`"""
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value):
        """Setter of `gt_instances`"""
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        """Deleter of `gt_instances`"""
        del self._gt_instances

    @property
    def features(self):
        """Setter of `features`"""
        return self._features

    @features.setter
    def features(self, value):
        """Setter of `features`"""
        self.set_field(value, '_features', dtype=InstanceData)

    @features.deleter
    def features(self):
        """Deleter of `features`"""
        del self._features

class PackActionInputs:
    """Pack the inputs data.

    Args:
        collect_keys (tuple[str], optional): The keys to be collected
            to ``packed_results['inputs']``. Defaults to ``
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
        algorithm_keys (Sequence[str]): The keys of custom elements to be used
            in the algorithm. Defaults to an empty tuple.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
            self,
            collect_keys: Optional[Tuple[str]] = None,
            meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                        'timestamp'),
            algorithm_keys: Sequence[str] = (),
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys
        self.algorithm_keys = algorithm_keys

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        packed_results = dict()
        if self.collect_keys is not None:
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results:
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs)
            elif 'heatmap_imgs' in results:
                heatmap_imgs = results['heatmap_imgs']
                packed_results['inputs'] = to_tensor(heatmap_imgs)
            elif 'keypoint' in results:
                keypoint = results['keypoint']
                packed_results['inputs'] = to_tensor(keypoint)
            elif 'audios' in results:
                audios = results['audios']
                packed_results['inputs'] = to_tensor(audios)
            elif 'text' in results:
                text = results['text']
                packed_results['inputs'] = to_tensor(text)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')

        data_sample = ActionDataSample()

        if 'gt_bboxes' in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            data_sample.gt_instances = instance_data

            if 'proposals' in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results['proposals']))

        if 'label' in results:
            data_sample.set_gt_label(results['label'])

        # Set custom algorithm keys
        for key in self.algorithm_keys:
            if key in results:
                data_sample.set_field(results[key], key)

        # Set meta keys
        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str
