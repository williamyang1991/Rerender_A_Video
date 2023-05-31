from typing import Optional

import json
import os
from src.video_util import get_frame_count, video_to_frame


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
                               sd_model: Optional[str] = None,
                               n_prompt: str = '',
                               control_type: str = 'canny',
                               control_strength=1,
                               seed: int = -1,
                               image_resolution: int = 512,
                               x0_strength: float = 0.1,
                               warp_step: float = 0.3,
                               ada_step: float = 0.8,
                               **kwargs):
        self.input_path = input_path
        self.output_path = output_path
        self.prompt = prompt
        self.work_dir = work_dir
        if work_dir is None:
            self.work_dir = os.path.basename(output_path)
        self.key_dir = os.path.join(self.work_dir, key_subdir)
        self.first_dir = os.path.join(self.work_dir, 'first')

        # Split video into frames
        assert os.path.isfile(input_path), 'The input must be a video'
        self.input_dir = os.path.join(self.work_dir, 'video')
        os.makedirs(self.input_dir, exist_ok=True)
        video_to_frame(input_path, self.input_dir, '%04d.png', False)

        self.frame_count = frame_count
        if frame_count is None:
            self.frame_count = get_frame_count(self.input_path)
        self.interval = interval
        self.sd_model = sd_model
        self.n_prompt = n_prompt
        self.control_type = control_type
        if self.control_type == 'canny':
            self.canny_low = kwargs.get("canny_low", 100)
            self.canny_high = kwargs.get("canny_high", 200)
        self.control_strength = control_strength
        self.seed = seed
        self.image_resolution = image_resolution
        self.x0_strength = x0_strength
        self.warp_step = warp_step
        self.ada_step = ada_step

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
        append_if_not_none('sd_model')
        append_if_not_none('n_prompt')
        append_if_not_none('control_type')
        if kwargs.get('control_type', '') == 'canny':
            append_if_not_none('canny_low')
            append_if_not_none('canny_high')
        append_if_not_none('control_strength')
        append_if_not_none('seed')
        append_if_not_none('image_resolution')
        append_if_not_none('x0_strength')
        append_if_not_none('warp_step')
        append_if_not_none('ada_step')
        self.create_from_parameters(**kwargs)
