#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from . import ClassyTransform, build_transforms, register_transform
from .util import ApplyTransformToKey, ImagenetConstants
import numpy as np
import math

class VideoConstants:
    """Constant variables related to the video classification.

    Use the same mean/std from image classification to enable the parameter
    inflation where parameters of 2D conv in image model can be inflated into
    3D conv in video model.

    MEAN: often used to be subtracted from pixel RGB value.
    STD: often used to divide the pixel RGB value after mean centering.
    SIZE_RANGE: a (min_size, max_size) tuple which denotes the range of
        size of the rescaled video clip.
    CROP_SIZE: the size of spatial cropping in the video clip.
    """

    MEAN = ImagenetConstants.MEAN  #
    STD = ImagenetConstants.STD
    SIZE_RANGE = (128, 160)
    CROP_SIZE = 112


def _get_rescaled_size(scale, h, w):
    if h < w:
        new_h = scale
        new_w = int(scale * w / h)
    else:
        new_w = scale
        new_h = int(scale * h / w)
    return new_h, new_w


class CustomCompose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, fps):
        for t in self.transforms:
            if isinstance(t, FrameDilation):
                img = t(img, fps)
            else:
                img = t(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

@register_transform("video_clip_random_resize_crop")
class VideoClipRandomResizeCrop(ClassyTransform):
    """A video clip transform that is often useful for trainig data.

    Given a size range, randomly choose a size. Rescale the clip so that
    its short edge equals to the chosen size. Then randomly crop the video
    clip with the specified size.
    Such training data augmentation is used in VGG net
    (https://arxiv.org/abs/1409.1556).
    Also see reference implementation `Kinetics.spatial_sampling` in SlowFast
        codebase.
    """

    def __init__(
        self,
        crop_size: Union[int, List[int]],
        size_range: List[int],
        interpolation_mode: str = "bilinear",
    ):
        """The constructor method of VideoClipRandomResizeCrop class.

        Args:
            crop_size: int or 2-tuple as the expected output crop_size (height, width)
            size_range: the min- and max size
            interpolation_mode: Default: "bilinear"

        """
        if isinstance(crop_size, tuple):
            assert len(crop_size) == 2, "crop_size should be tuple (height, width)"
            self.crop_size = crop_size
        else:
            self.crop_size = (crop_size, crop_size)

        self.interpolation_mode = interpolation_mode
        self.size_range = size_range

    def __call__(self, clip):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        # clip size: C x T x H x W
        rand_size = random.randint(self.size_range[0], self.size_range[1])
        new_h, new_w = _get_rescaled_size(rand_size, clip.size()[2], clip.size()[3])
        clip = torch.nn.functional.interpolate(
            clip, size=(new_h, new_w), mode=self.interpolation_mode
        )
        assert (
            self.crop_size[0] <= new_h and self.crop_size[1] <= new_w
        ), "crop size can not be larger than video frame size"

        i = random.randint(0, new_h - self.crop_size[0])
        j = random.randint(0, new_w - self.crop_size[1])
        clip = clip[:, :, i : i + self.crop_size[0], j : j + self.crop_size[1]]
        return clip


@register_transform("frame_dilation")
class FrameDilation(ClassyTransform):
    """A video clip transform that is often useful for trainig data.

    Given a size range, randomly choose a size. Rescale the clip so that
    its short edge equals to the chosen size. Then randomly crop the video
    clip with the specified size.
    Such training data augmentation is used in VGG net
    (https://arxiv.org/abs/1409.1556).
    Also see reference implementation `Kinetics.spatial_sampling` in SlowFast
        codebase.
    """

    def __init__(
        self,
        target_fps: int = 1,
    ):
        """The constructor method of VideoClipRandomResizeCrop class.

        Args:
            crop_size: int or 2-tuple as the expected output crop_size (height, width)
            size_range: the min- and max size
            interpolation_mode: Default: "bilinear"

        """

        self.target_fps = target_fps

    def __call__(self, clip, fps=1):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        frame_dilation = int(fps/float(self.target_fps))
        offset = np.random.randint(frame_dilation)
        index = list(range(0+offset, clip.shape[1]+offset, frame_dilation))
        print('index: {}'.format(index))
        clip = clip[:, index, :, :]
        return clip

def crop(img, top, left, height, width):
    # type: (Tensor, int, int, int, int) -> Tensor
    """Crop the given Image Tensor.

    Args:
        img (Tensor): Image to be cropped in the form [C, H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        Tensor: Cropped image.
    """


    return img[..., top:top + height, left:left + width]

def center_crop(img, output_size):
    # type: (Tensor, BroadcastingList2[int]) -> Tensor
    """Crop the Image Tensor and resize it to desired size.

    Args:
        img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions

    Returns:
            Tensor: Cropped image.
    """

    _, _, image_height, image_width = img.size()
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return crop(img, crop_top, crop_left, crop_height, crop_width)

def spatial_shift_crop_list(size, video, spatial_shift_pos, boxes=None):
    """
    Perform left, center, or right crop of the given list of images.
    Args:
        size (int): size to crop.
        image (list): ilist of images to perform short side scale. Dimension is
            `height` x `width` x `channel` or `channel` x `height` x `width`.
        spatial_shift_pos (int): option includes 0 (left), 1 (middle), and
            2 (right) crop.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (ndarray): the cropped list of images with dimension of
            `height` x `width` x `channel`.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """

    assert spatial_shift_pos in [0, 1, 2]

    height = video.shape[-2]
    width = video.shape[-1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = video[:, :, y_offset : y_offset + size, x_offset : x_offset + size]
    assert cropped.shape[-1] == size, "Image height not cropped properly"
    assert cropped.shape[-2] == size, "Image width not cropped properly"

    # if boxes is not None:
    #     for i in range(len(boxes)):
    #         boxes[i][:, [0, 2]] -= x_offset
    #         boxes[i][:, [1, 3]] -= y_offset
    return cropped

@register_transform("three_crop")
class ThreeCrop(ClassyTransform):
    """A video clip transform that is often useful for testing data.

    Given an input size, rescale the clip so that its short edge equals to
    the input size while aspect ratio is preserved.
    """

    def __init__(self, size: int):
        """The constructor method of VideoClipResize class.

        Args:
            size: input size
            interpolation_mode: Default: "bilinear". See valid values in
                (https://pytorch.org/docs/stable/nn.functional.html#torch.nn.
                functional.interpolate)

        """
        self.size = size

    def __call__(self, clip):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        # channel, frame, image_height, image_width = clip.size()
        # # print(clip.size())
        # crop_height, crop_width = self.size, self.size
        # if crop_width > image_width or crop_height > image_height:
        #     msg = "Requested crop size {} is bigger than input size {}"
        #     raise ValueError(msg.format(self.size, (image_height, image_width)))
        #
        # left = crop(clip, 0, 0, crop_width, crop_height)
        # right = crop(clip, 0, image_width-crop_width, crop_width, crop_height)
        # center = center_crop(clip, (crop_height, crop_width))

        left = spatial_shift_crop_list(self.size, clip, 0, boxes=None)
        center = spatial_shift_crop_list(self.size, clip, 1, boxes=None)
        right = spatial_shift_crop_list(self.size, clip, 2, boxes=None)

        out = torch.cat((left, center, right))
        assert out.shape[0] == 9
        # return [left, center, right]
        return out


@register_transform("video_clip_resize")
class VideoClipResize(ClassyTransform):
    """A video clip transform that is often useful for testing data.

    Given an input size, rescale the clip so that its short edge equals to
    the input size while aspect ratio is preserved.
    """

    def __init__(self, size: int, interpolation_mode: str = "bilinear"):
        """The constructor method of VideoClipResize class.

        Args:
            size: input size
            interpolation_mode: Default: "bilinear". See valid values in
                (https://pytorch.org/docs/stable/nn.functional.html#torch.nn.
                functional.interpolate)

        """
        self.interpolation_mode = interpolation_mode
        self.size = size

    def __call__(self, clip):
        """Callable function which applies the tranform to the input clip.

        Args:
            clip (torch.Tensor): input clip tensor

        """
        # clip size: C x T x H x W
        if not min(clip.size()[2], clip.size()[3]) == self.size:
            new_h, new_w = _get_rescaled_size(self.size, clip.size()[2], clip.size()[3])
            clip = torch.nn.functional.interpolate(
                clip, size=(new_h, new_w), mode=self.interpolation_mode
            )
        return clip


@register_transform("video_tuple_to_map_transform")
class VideoTupleToMapTransform(ClassyTransform):
    """A video transform which maps video data from tuple to dict.

    It takes a sample of the form (video, audio, target) and returns a sample of
    the form {"input": {"video" video, "audio": audio}, "target": target}. If
    the sample is a map with these keys already present, it will pass the sample
    through.

    It's particularly useful for remapping torchvision samples which are
    tuples of the form (video, audio, target).
    """

    def __call__(self, sample):
        """Callable function which applies the tranform to the input sample data.

        Args:
            sample: input sample data that will undergo the transform

        """
        # If sample is a map and already has input / target keys, pass through
        if isinstance(sample, dict):
            assert "input" in sample and "target" in sample, (
                "Input to tuple to map transform must be a tuple of length 3 "
                "or a dict with keys 'input' and 'target'"
            )
            assert (
                "video" in sample["input"] and "audio" in sample["input"]
            ), "Input data must include video / audio fields"
            return sample

        # Should be a tuple (or other sequential) of length 3, transform to map
        assert len(sample) == 3, "Sequential must be length 3 for conversion"
        video, audio, target = sample
        output_sample = {"input": {"video": video, "audio": audio}, "target": target}
        return output_sample


@register_transform("video_default_augment")
class VideoDefaultAugmentTransform(ClassyTransform):
    """This is the default video transform with data augmentation which is useful for
    training.

    It sequentially prepares a torch.Tensor of video data, randomly
    resizes the video clip, takes a random spatial cropping, randomly flips the
    video clip horizontally, and normalizes the pixel values by mean subtraction
    and standard deviation division.

    """

    def __init__(
        self,
        crop_size: Union[int, List[int]] = VideoConstants.CROP_SIZE,
        size_range: List[int] = VideoConstants.SIZE_RANGE,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        """The constructor method of VideoDefaultAugmentTransform class.

        Args:
            crop_size: expected output crop_size (height, width)
            size_range : a 2-tuple denoting the min- and max size
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """

        self._transform = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                # TODO(zyan3): migrate VideoClipRandomResizeCrop to TorchVision
                VideoClipRandomResizeCrop(crop_size, size_range),
                transforms_video.RandomHorizontalFlipVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        """Apply the default transform with data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)

@register_transform("video_default_augment_frame_dilation")
class VideoDefaultAugmentFrameTransform(ClassyTransform):
    """This is the default video transform with data augmentation which is useful for
    training.

    It sequentially prepares a torch.Tensor of video data, randomly
    resizes the video clip, takes a random spatial cropping, randomly flips the
    video clip horizontally, and normalizes the pixel values by mean subtraction
    and standard deviation division.

    """

    def __init__(
        self,
        crop_size: Union[int, List[int]] = VideoConstants.CROP_SIZE,
        size_range: List[int] = VideoConstants.SIZE_RANGE,
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
        target_fps: int = 5,
    ):
        """The constructor method of VideoDefaultAugmentTransform class.

        Args:
            crop_size: expected output crop_size (height, width)
            size_range : a 2-tuple denoting the min- and max size
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """

        self._transform = transforms.Compose(
            [
                transforms_video.ToTensorVideo(),
                FrameDilation(target_fps),
                # TODO(zyan3): migrate VideoClipRandomResizeCrop to TorchVision
                VideoClipRandomResizeCrop(crop_size, size_range),
                transforms_video.RandomHorizontalFlipVideo(),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        """Apply the default transform with data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)


@register_transform("video_default_no_augment")
class VideoDefaultNoAugmentTransform(ClassyTransform):
    """This is the default video transform without data augmentation which is useful
    for testing.

    It sequentially prepares a torch.Tensor of video data, resize the
    video clip to have the specified short edge, and normalize the pixel values
    by mean subtraction and standard deviation division.

    """

    def __init__(
        self,
        size: int = VideoConstants.SIZE_RANGE[0],
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        """The constructor method of VideoDefaultNoAugmentTransform class.

        Args:
            size: the short edge of rescaled video clip
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """
        self._transform = transforms.Compose(
            # At testing stage, central cropping is not used because we
            # conduct fully convolutional-style testing
            [
                transforms_video.ToTensorVideo(),
                # TODO(zyan3): migrate VideoClipResize to TorchVision
                VideoClipResize(size),
                transforms_video.NormalizeVideo(mean=mean, std=std),
            ]
        )

    def __call__(self, video):
        """Apply the default transform without data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)


@register_transform("video_default_no_augment_three_crop")
class VideoDefaultNoAugmentTransformThreeCrop(ClassyTransform):
    """This is the default video transform without data augmentation which is useful
    for testing.

    It sequentially prepares a torch.Tensor of video data, resize the
    video clip to have the specified short edge, and normalize the pixel values
    by mean subtraction and standard deviation division.

    """

    def __init__(
        self,
        size: int = VideoConstants.SIZE_RANGE[0],
        mean: List[float] = VideoConstants.MEAN,
        std: List[float] = VideoConstants.STD,
    ):
        """The constructor method of VideoDefaultNoAugmentTransform class.

        Args:
            size: the short edge of rescaled video clip
            mean: a 3-tuple denoting the pixel RGB mean
            std: a 3-tuple denoting the pixel RGB standard deviation

        """
        self._transform = transforms.Compose(
            # At testing stage, central cropping is not used because we
            # conduct fully convolutional-style testing
            [
                transforms_video.ToTensorVideo(),
                VideoClipResize(size),
                # TODO(zyan3): migrate VideoClipResize to TorchVision
                transforms_video.NormalizeVideo(mean=mean, std=std),
                ThreeCrop(size),
            ]
        )

    def __call__(self, video):
        """Apply the default transform without data augmentation to video.

        Args:
            video: input video that will undergo the transform

        """
        return self._transform(video)


@register_transform("dummy_audio_transform")
class DummyAudioTransform(ClassyTransform):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        """The constructor method of DummyAudioTransform class.
        """

        pass

    def __call__(self, _audio):
        """Callable function which applies the tranform to the input audio data.

        Args:
            audio: input audio data that will undergo the dummy transform

        """
        return torch.zeros(0, 1, dtype=torch.float)


class ClassyVideoGenericTransform(object):
    """This is a generic video transform which includes both video transform
    and audio transform.
    """

    def __init__(
        self,
        config: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        split: str = "train",
    ):
        """The constructor method of ClassyVideoGenericTransform class.

        Args:
            config: If provided, it is a dict where key is the data modality, and
                value is a dict specifying the transform config
            split: the split of the data to which the transform will be applied
        """
        self.transforms = {
            "video": VideoDefaultAugmentTransform()
            if split == "train"
            else VideoDefaultNoAugmentTransform(),
            "audio": DummyAudioTransform(),
        }
        if config is not None:
            for mode, modal_config in config.items():
                assert mode in ["video", "audio"], (
                    "unknown video data modality %s" % mode
                )
                self.transforms[mode] = build_transforms(modal_config)

    def __call__(self, video: Dict):
        """Callable function which applies the tranform to the input video data.

        Args:
            video: input video data that will undergo the transform

        """
        assert isinstance(video, dict), "video data is expected be a dict"
        for mode, modal_data in video.items():
            if mode in self.transforms:
                video[mode] = self.transforms[mode](modal_data)
        if video["video"].shape[0] == 9:
            video['video'] = torch.stack((video["video"][:3,:,:,:], video["video"][3:6,:,:,:], video["video"][6:,:,:,:]))
        return video


DEFAULT_KEY_MAP = VideoTupleToMapTransform()


def build_video_field_transform_default(
    config: Optional[Dict[str, List[Dict[str, Any]]]],
    split: str = "train",
    key: str = "input",
    key_map_transform: Optional[Callable] = DEFAULT_KEY_MAP,
) -> Callable:
    """Returns transform that first maps sample to video keys, then
    returns a transform on the specified key in dict.

    Converts tuple (list, etc) sample to dict with input / target keys.
    For a dict sample, verifies that dict has input / target keys.
    For all other samples throws.

    Args:
        config: If provided, it is a dict where key is the data modality, and
            value is a dict specifying the transform config
        split: the split of the data to which the transform will be applied
        key: the key in data sample of type dict whose corresponding value will
            undergo the transform
        key_map_transform: If provided, it is a transform which maps sample of type
            tuple to sample of type dict. See default value VideoTupleToMapTransform()
            as an example

    """
    transform = ApplyTransformToKey(ClassyVideoGenericTransform(config, split), key=key)
    if key_map_transform is None:
        return transform

    return transforms.Compose([key_map_transform, transform])
