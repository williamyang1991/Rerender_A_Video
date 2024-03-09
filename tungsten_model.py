"""
Tungsten model definition.

Before start building,
  1) Download SD weights in ./models directory.
  2) Replace global variables `SD_MODEL_PATH` with your SD weight file name
  3) (Optional) Update global variables `DEFAULT_ADDED_PROMPT` and `DEFAULT_NEGATIVE_PROMPT`
"""

import os
import random
import shutil
import warnings
from typing import List

import cv2
import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from tungstenkit import BaseIO, Field, Option, Video, define_model

import src.import_util  # noqa: F401
from deps.ControlNet.annotator.canny import CannyDetector
from deps.ControlNet.annotator.hed import HEDdetector
from deps.ControlNet.annotator.util import HWC3
from deps.ControlNet.cldm.cldm import ControlLDM
from deps.ControlNet.cldm.model import create_model, load_state_dict
from deps.gmflow.gmflow.gmflow import GMFlow
from src.config import RerenderConfig
from src.ddim_v_hacked import DDIMVSampler
from src.freeu import freeu_forward
from src.img_util import find_flat_region, numpy2tensor
from src.video_util import prepare_frames

warnings.filterwarnings("ignore")


SD_MODEL_FILE_NAME = "realisticVisionV20_v20.safetensors"

DEFAULT_ADDED_PROMPT = "best quality, extremely detailed"
DEFAULT_NEGATIVE_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, "
    "cropped, worst quality, low quality"
)


class Input(BaseIO):
    input_video: Video = Field(description="Input video to be rerendered")
    prompt: str = Field(description="Describe what you want to see in the output video")

    seed: int = Option(-1, description="Random seed. Set as -1 to randomize.")
    frame_resolution: int = Option(
        512,
        description="Frame resolution of the output video",
        ge=256,
        le=768,
    )
    denoising_strength: float = Option(
        0.75,
        description="0: fully recover the input / 1.05: fully rerender the input",
        ge=0.0,
        le=1.05,
    )
    preserve_color: bool = Option(
        True,
        description="Keep the color of the input video",
    )
    sampling_steps: int = Option(
        20,
        ge=1,
        le=50,
    )
    cfg_scale: float = Option(
        7.5,
        ge=0.1,
        le=30,
    )
    controlnet_strength: float = Option(
        0.8,
        ge=0.0,
        le=2.0,
    )
    control_type: str = Option("canny", choices=["canny", "HED"])
    canny_low_threshold: int = Option(100, ge=0, le=255)
    canny_high_threshold: int = Option(200, ge=0, le=255)
    key_frame_frequency: int = Option(
        10,
        description="Uniformly sample the key frames every K frames",
        ge=1,
        le=120,
    )
    left_crop_length: int = Option(0)
    right_crop_length: int = Option(0)
    top_crop_length: int = Option(0)
    bottom_crop_length: int = Option(0)
    added_prompt: str = Option(DEFAULT_ADDED_PROMPT)
    negative_prompt: str = Option(DEFAULT_NEGATIVE_PROMPT)

    def to_rerender_config(self, *, output_path: str):
        cfg = RerenderConfig()
        cfg.create_from_parameters(
            input_path=str(self.input_video.path),
            output_path=output_path,
            prompt=self.prompt,
            interval=self.key_frame_frequency,
            crop=(
                self.left_crop_length,
                self.right_crop_length,
                self.top_crop_length,
                self.bottom_crop_length,
            ),
            sd_model=SD_MODEL_FILE_NAME,
            a_prompt=self.added_prompt,
            n_prompt=self.negative_prompt,
            ddim_steps=self.sampling_steps,
            scale=self.cfg_scale,
            control_type=self.control_type,
            control_strength=self.controlnet_strength,
            seed=self.seed,
            image_resolution=self.frame_resolution,
            x0_strength=self.denoising_strength,
            warp_period=(0.0, 0.1),
            mask_period=(0.5, 0.8),
            ada_period=(0.8, 0.1),
            cross_period=(0, 1),
            smooth_boundary=True,
            style_update_freq=1,
        )
        return cfg


class Output(BaseIO):
    output_video: Video


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    include_files=[
        "blender",
        "config",
        "deps",
        "flow",
        "src",
        "*.py",
        os.path.join("models", SD_MODEL_FILE_NAME),
    ],
    python_version="3.8",
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "build-essential",
    ],
    python_packages=[
        "torch==2.0.0",
        "torchvision==0.15.1",
        "numpy==1.23.1",
        "gradio==3.44.4",
        "albumentations==1.3.0",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.2",
        "pytorch-lightning==1.5.0",
        "omegaconf==2.1.1",
        "test-tube>=0.7.5",
        "streamlit==1.12.1",
        "einops==0.3.0",
        "transformers==4.19.2",
        "webdataset==0.2.5",
        "kornia==0.6",
        "open_clip_torch==2.0.2",
        "invisible-watermark>=0.1.5",
        "streamlit-drawable-canvas==0.8.0",
        "torchmetrics==0.6.0",
        "timm==0.6.12",
        "addict==2.4.0",
        "yapf==0.32.0",
        "prettytable==3.6.0",
        "safetensors==0.2.7",
        "basicsr==1.4.2",
        "blendmodes",
        "numba==0.57.0",
        "opencv-python==4.8.1.78",
    ],
    cuda_version="11.8",
    force_install_system_cuda=True,
    batch_size=1,
)
class RerenderModel:
    @staticmethod
    def post_build():
        """Download model data"""
        os.system("python install.py")
        create_model("./deps/ControlNet/models/cldm_v15.yaml")

    def setup(self):
        """Load model weights"""
        self.model: ControlLDM = create_model(
            "./deps/ControlNet/models/cldm_v15.yaml"
        ).cuda()
        self.control_type = "canny"
        self.model.load_state_dict(
            load_state_dict("./models/control_sd15_canny.pth", location="cuda")
        )
        sd_model_path = os.path.join("models", SD_MODEL_FILE_NAME)
        sd_model_ext = os.path.splitext(SD_MODEL_FILE_NAME)[-1]
        if sd_model_ext == ".safetensors":
            self.model.load_state_dict(load_file(sd_model_path), strict=False)
        elif sd_model_ext == ".ckpt" or sd_model_ext == ".pth":
            self.model.load_state_dict(
                torch.load(sd_model_path)["state_dict"], strict=False
            )
        else:
            raise RuntimeError(f"Unknown checkpoint extension: {sd_model_ext}")
        self.model.first_stage_model.load_state_dict(
            torch.load("./models/vae-ft-mse-840000-ema-pruned.ckpt")["state_dict"],
            strict=False,
        )
        freeu_args = (1.1, 1.2, 1.0, 0.2)
        self.model.model.diffusion_model.forward = freeu_forward(
            self.model.model.diffusion_model, *freeu_args
        )

        self.ddim_v_sampler = DDIMVSampler(self.model)

        self.flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type="swin",
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to("cuda")
        flow_model_checkpoint = torch.load(
            "models/gmflow_sintel-0c07dcb3.pth",
            map_location=lambda storage, loc: storage,
        )
        flow_model_weights = (
            flow_model_checkpoint["model"]
            if "model" in flow_model_checkpoint
            else flow_model_checkpoint
        )
        self.flow_model.load_state_dict(flow_model_weights, strict=False)
        self.flow_model.eval()

    def predict(self, inputs: List[Input]) -> List[Output]:
        """Run a prediction"""
        from flow.flow_utils import get_warped_and_mask
        from rerender import apply_color_correction, postprocess, setup_color_correction
        from src.controller import AttentionControl

        input = inputs[0]  # batch_size == 1

        if self.control_type != input.control_type:
            self._load_controlnet_weights(input.control_type)
        self._set_detector(
            input.control_type, input.canny_low_threshold, input.canny_high_threshold
        )

        if os.path.exists("results"):
            shutil.rmtree("results")
        cfg = input.to_rerender_config(output_path="results/output.mp4")
        if cfg.frame_count > 102:
            print("Input video is too long. Use only first 102 frames.")
            cfg.frame_count = 102

        blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
        totensor = T.PILToTensor()

        prepare_frames(
            cfg.input_path,
            cfg.input_dir,
            cfg.image_resolution,
            cfg.crop,
            cfg.use_limit_device_resolution,
        )

        num_samples = 1
        ddim_steps = cfg.ddim_steps
        scale = cfg.scale

        seed = cfg.seed
        if seed == -1:
            seed = random.randint(0, 65535)
        eta = 0.0

        prompt = cfg.prompt
        a_prompt = cfg.a_prompt
        n_prompt = cfg.n_prompt
        prompt = prompt + ", " + a_prompt

        style_update_freq = cfg.style_update_freq
        pixelfusion = True
        color_preserve = cfg.color_preserve

        x0_strength = 1 - cfg.x0_strength
        mask_period = cfg.mask_period
        firstx0 = True
        controller = AttentionControl(
            cfg.inner_strength,
            cfg.mask_period,
            cfg.cross_period,
            cfg.ada_period,
            cfg.warp_period,
            cfg.loose_cfattn,
        )

        imgs = sorted(os.listdir(cfg.input_dir))
        imgs = [os.path.join(cfg.input_dir, img) for img in imgs]
        if cfg.frame_count >= 0:
            imgs = imgs[: cfg.frame_count]

        with torch.no_grad():
            frame = cv2.imread(imgs[0])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = HWC3(frame)
            H, W, C = img.shape

            img_ = numpy2tensor(img)
            # if color_preserve:
            #     img_ = numpy2tensor(img)
            # else:
            #     img_ = apply_color_correction(color_corrections,
            #                                   Image.fromarray(img))
            #     img_ = totensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
            encoder_posterior = self.model.encode_first_stage(img_.cuda())
            x0 = self.model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = self.detector(img)
            detected_map = HWC3(detected_map)
            # For visualization
            detected_img = 255 - detected_map

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, "b h w c -> b c h w").clone()
            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([prompt] * num_samples)
                ],
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([n_prompt] * num_samples)
                ],
            }
            shape = (4, H // 8, W // 8)

            controller.set_task("initfirst")
            seed_everything(seed)
            samples, _ = self.ddim_v_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=x0_strength,
            )
            x_samples = self.model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img
            first_result = pre_result
            first_img = pre_img

            x_samples = (
                (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .clip(0, 255)
                .astype(np.uint8)
            )
        color_corrections = setup_color_correction(Image.fromarray(x_samples[0]))
        Image.fromarray(x_samples[0]).save(os.path.join(cfg.first_dir, "first.jpg"))
        cv2.imwrite(os.path.join(cfg.first_dir, "first_edge.jpg"), detected_img)

        for i in range(0, min(len(imgs), cfg.frame_count) - 1, cfg.interval):
            cid = i + 1
            print(f"Key frame: {cid}/{cfg.frame_count}")
            if cid <= (len(imgs) - 1):
                frame = cv2.imread(imgs[cid])
            else:
                frame = cv2.imread(imgs[len(imgs) - 1])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = HWC3(frame)

            if color_preserve:
                img_ = numpy2tensor(img)
            else:
                img_ = apply_color_correction(color_corrections, Image.fromarray(img))
                img_ = totensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
            encoder_posterior = self.model.encode_first_stage(img_.cuda())
            x0 = self.model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = self.detector(img)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, "b h w c -> b c h w").clone()
            cond["c_concat"] = [control]
            un_cond["c_concat"] = [control]

            image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
            image2 = torch.from_numpy(img).permute(2, 0, 1).float()
            warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(
                self.flow_model, image1, image2, pre_result, False
            )
            blend_mask_pre = blur(
                F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4)
            )
            blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

            image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
            warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(
                self.flow_model, image1, image2, first_result, False
            )
            blend_mask_0 = blur(
                F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4)
            )
            blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

            if firstx0:
                mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
                controller.set_warp(
                    F.interpolate(
                        bwd_flow_0 / 8.0, scale_factor=1.0 / 8, mode="bilinear"
                    ),
                    mask,
                )
            else:
                mask = 1 - F.max_pool2d(blend_mask_pre, kernel_size=8)
                controller.set_warp(
                    F.interpolate(
                        bwd_flow_pre / 8.0, scale_factor=1.0 / 8, mode="bilinear"
                    ),
                    mask,
                )

            controller.set_task("keepx0, keepstyle")
            seed_everything(seed)
            samples, intermediates = self.ddim_v_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=x0_strength,
            )
            direct_result = self.model.decode_first_stage(samples)

            if not pixelfusion:
                pre_result = direct_result
                pre_img = img
                viz = (
                    (
                        einops.rearrange(direct_result, "b c h w -> b h w c") * 127.5
                        + 127.5
                    )
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

            else:
                blend_results = (
                    1 - blend_mask_pre
                ) * warped_pre + blend_mask_pre * direct_result
                blend_results = (
                    1 - blend_mask_0
                ) * warped_0 + blend_mask_0 * blend_results

                bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
                blend_mask = blur(
                    F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4)
                )
                blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

                encoder_posterior = self.model.encode_first_stage(blend_results)
                xtrg = self.model.get_first_stage_encoding(
                    encoder_posterior
                ).detach()  # * mask
                blend_results_rec = self.model.decode_first_stage(xtrg)
                encoder_posterior = self.model.encode_first_stage(blend_results_rec)
                xtrg_rec = self.model.get_first_stage_encoding(
                    encoder_posterior
                ).detach()
                xtrg_ = xtrg + 1 * (xtrg - xtrg_rec)  # * mask
                blend_results_rec_new = self.model.decode_first_stage(xtrg_)
                tmp = (
                    abs(blend_results_rec_new - blend_results).mean(
                        dim=1, keepdims=True
                    )
                    > 0.25
                ).float()
                mask_x = F.max_pool2d(
                    (
                        F.interpolate(tmp, scale_factor=1 / 8.0, mode="bilinear") > 0
                    ).float(),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )

                mask = 1 - F.max_pool2d(1 - blend_mask, kernel_size=8)  # * (1-mask_x)

                if cfg.smooth_boundary:
                    noise_rescale = find_flat_region(mask)
                else:
                    noise_rescale = torch.ones_like(mask)
                masks = []
                for i in range(ddim_steps):
                    if (
                        i <= ddim_steps * mask_period[0]
                        or i >= ddim_steps * mask_period[1]
                    ):
                        masks += [None]
                    else:
                        masks += [mask * cfg.mask_strength]

                # mask 3
                # xtrg = ((1-mask_x) *
                #         (xtrg + xtrg - xtrg_rec) + mask_x * samples) * mask
                # mask 2
                # xtrg = (xtrg + 1 * (xtrg - xtrg_rec)) * mask
                xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1

                tasks = "keepstyle, keepx0"
                if not firstx0:
                    tasks += ", updatex0"
                if i % style_update_freq == 0:
                    tasks += ", updatestyle"
                controller.set_task(tasks, 1.0)

                seed_everything(seed)
                samples, _ = self.ddim_v_sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=eta,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=un_cond,
                    controller=controller,
                    x0=x0,
                    strength=x0_strength,
                    xtrg=xtrg,
                    mask=masks,
                    noise_rescale=noise_rescale,
                )
                x_samples = self.model.decode_first_stage(samples)
                pre_result = x_samples
                pre_img = img

                viz = (
                    (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                    .cpu()
                    .numpy()
                    .clip(0, 255)
                    .astype(np.uint8)
                )

            Image.fromarray(viz[0]).save(os.path.join(cfg.key_dir, f"{cid:04d}.png"))
            print()

        print("Postprocessing...")
        torch.cuda.empty_cache()
        postprocess(cfg, ne=False, max_process=6, ps=True, tmp=True)

        return [Output(output_video=Video.from_path("results/output.mp4"))]

    def _set_detector(self, control_type: str, canny_low: int, canny_high: int):
        if control_type == "HED":
            self.detector = HEDdetector()
        elif control_type == "canny":
            canny_detector = CannyDetector()
            low_threshold = canny_low
            high_threshold = canny_high

            def apply_canny(x):
                return canny_detector(x, low_threshold, high_threshold)

            self.detector = apply_canny
        else:
            raise RuntimeError(f"Unsupported control_type: {control_type}")

    def _load_controlnet_weights(self, control_type: str):
        if control_type == "HED":
            self.model.load_state_dict(
                load_state_dict("./models/control_sd15_hed.pth", location="cuda")
            )
        elif control_type == "canny":
            self.model.load_state_dict(
                load_state_dict("./models/control_sd15_canny.pth", location="cuda")
            )
        else:
            raise RuntimeError(f"Unsupported control_type: {control_type}")
