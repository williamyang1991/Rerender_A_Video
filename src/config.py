import json
import os
from typing import Optional, Sequence, Tuple

from src.video_util import get_frame_count


class RerenderConfig:

    def __init__(self):
        ...

    def create_from_parameters(self,
                               input_path: str,
                               output_path: str,
                               prompt: str,
                               work_dir: Optional[str] = None,
                               key_subdir: str = 'keys',
                               frame_count: Optional[int] = None,
                               interval: int = 10,
                               crop: Sequence[int] = (0, 0, 0, 0),
                               sd_model: Optional[str] = None,
                               a_prompt: str = '',
                               n_prompt: str = '',
                               ddim_steps=20,
                               scale=7.5,
                               control_type: str = 'HED',
                               control_strength=1,
                               seed: int = -1,
                               image_resolution: int = 512,
                               use_limit_device_resolution: bool = False,
                               x0_strength: float = -1,
                               style_update_freq: int = 10,
                               cross_period: Tuple[float, float] = (0, 1),
                               warp_period: Tuple[float, float] = (0, 0.1),
                               mask_period: Tuple[float, float] = (0.5, 0.8),
                               ada_period: Tuple[float, float] = (1.0, 1.0),
                               mask_strength: float = 0.5,
                               inner_strength: float = 0.9,
                               smooth_boundary: bool = True,
                               color_preserve: bool = True,
                               loose_cfattn: bool = False,
                               freeu_args: Tuple[int] = (1, 1, 1, 1),
                               **kwargs):
        self.input_path = input_path
        self.output_path = output_path
        self.prompt = prompt
        self.work_dir = work_dir
        if work_dir is None:
            self.work_dir = os.path.dirname(output_path)
        self.key_dir = os.path.join(self.work_dir, key_subdir)
        self.first_dir = os.path.join(self.work_dir, 'first')

        # Split video into frames
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f'Cannot find video file {input_path}')
        self.input_dir = os.path.join(self.work_dir, 'video')

        self.frame_count = frame_count
        if frame_count is None:
            self.frame_count = get_frame_count(self.input_path)
        self.interval = interval
        self.crop = crop
        self.sd_model = sd_model
        self.a_prompt = a_prompt
        self.n_prompt = n_prompt
        self.ddim_steps = ddim_steps
        self.scale = scale
        self.control_type = control_type
        if self.control_type == 'canny':
            self.canny_low = kwargs.get('canny_low', 100)
            self.canny_high = kwargs.get('canny_high', 200)
        else:
            self.canny_low = None
            self.canny_high = None
        self.control_strength = control_strength
        self.seed = seed
        self.image_resolution = image_resolution
        self.use_limit_device_resolution = use_limit_device_resolution
        self.x0_strength = x0_strength
        self.style_update_freq = style_update_freq
        self.cross_period = cross_period
        self.mask_period = mask_period
        self.warp_period = warp_period
        self.ada_period = ada_period
        self.mask_strength = mask_strength
        self.inner_strength = inner_strength
        self.smooth_boundary = smooth_boundary
        self.color_preserve = color_preserve
        self.loose_cfattn = loose_cfattn
        self.freeu_args = freeu_args

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.key_dir, exist_ok=True)
        os.makedirs(self.first_dir, exist_ok=True)

    def create_from_path(self, cfg_path: str):
        with open(cfg_path, 'r') as fp:
            cfg = json.load(fp)
        kwargs = dict()

        def append_if_not_none(key):
            value = cfg.get(key, None)
            if value is not None:
                kwargs[key] = value

        kwargs['input_path'] = cfg['input']
        kwargs['output_path'] = cfg['output']
        kwargs['prompt'] = cfg['prompt']
        append_if_not_none('work_dir')
        append_if_not_none('key_subdir')
        append_if_not_none('frame_count')
        append_if_not_none('interval')
        append_if_not_none('crop')
        append_if_not_none('sd_model')
        append_if_not_none('a_prompt')
        append_if_not_none('n_prompt')
        append_if_not_none('ddim_steps')
        append_if_not_none('scale')
        append_if_not_none('control_type')
        if kwargs.get('control_type', '') == 'canny':
            append_if_not_none('canny_low')
            append_if_not_none('canny_high')
        append_if_not_none('control_strength')
        append_if_not_none('seed')
        append_if_not_none('image_resolution')
        append_if_not_none('use_limit_device_resolution')
        append_if_not_none('x0_strength')
        append_if_not_none('style_update_freq')
        append_if_not_none('cross_period')
        append_if_not_none('warp_period')
        append_if_not_none('mask_period')
        append_if_not_none('ada_period')
        append_if_not_none('mask_strength')
        append_if_not_none('inner_strength')
        append_if_not_none('smooth_boundary')
        append_if_not_none('color_perserve')
        append_if_not_none('loose_cfattn')
        append_if_not_none('freeu_args')
        self.create_from_parameters(**kwargs)

    @property
    def use_warp(self):
        return self.warp_period[0] <= self.warp_period[1]

    @property
    def use_mask(self):
        return self.mask_period[0] <= self.mask_period[1]

    @property
    def use_ada(self):
        return self.ada_period[0] <= self.ada_period[1]
