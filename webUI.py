import os
import shutil
from enum import Enum

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from blendmodes.blend import BlendType, blendLayers
from PIL import Image
from pytorch_lightning import seed_everything
from safetensors.torch import load_file
from skimage import exposure

import src.import_util  # noqa: F401
from deps.ControlNet.annotator.canny import CannyDetector
from deps.ControlNet.annotator.hed import HEDdetector
from deps.ControlNet.annotator.util import HWC3
from deps.ControlNet.cldm.model import create_model, load_state_dict
from deps.gmflow.gmflow.gmflow import GMFlow
from flow.flow_utils import get_warped_and_mask
from sd_model_cfg import model_dict
from src.config import RerenderConfig
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.freeu import freeu_forward
from src.img_util import find_flat_region, numpy2tensor
from src.video_util import (frame_to_video, get_fps, get_frame_count,
                            prepare_frames)

inversed_model_dict = dict()
for k, v in model_dict.items():
    inversed_model_dict[v] = k

to_tensor = T.PILToTensor()
blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))


class ProcessingState(Enum):
    NULL = 0
    FIRST_IMG = 1
    KEY_IMGS = 2


class GlobalState:

    def __init__(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        self.detector_type = None
        self.detector = None
        self.controller = None
        self.processing_state = ProcessingState.NULL
        flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to('cuda')

        checkpoint = torch.load('models/gmflow_sintel-0c07dcb3.pth',
                                map_location=lambda storage, loc: storage)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()
        self.flow_model = flow_model

    def update_controller(self, inner_strength, mask_period, cross_period,
                          ada_period, warp_period, loose_cfattn):
        self.controller = AttentionControl(inner_strength,
                                           mask_period,
                                           cross_period,
                                           ada_period,
                                           warp_period,
                                           loose_cfatnn=loose_cfattn)

    def update_sd_model(self, sd_model, control_type, freeu_args):
        if sd_model == self.sd_model:
            return
        self.sd_model = sd_model
        model = create_model('./deps/ControlNet/models/cldm_v15.yaml').cpu()
        if control_type == 'HED':
            model.load_state_dict(
                load_state_dict('./models/control_sd15_hed.pth',
                                location='cuda'))
        elif control_type == 'canny':
            model.load_state_dict(
                load_state_dict('./models/control_sd15_canny.pth',
                                location='cuda'))
        model = model.cuda()
        sd_model_path = model_dict[sd_model]
        if len(sd_model_path) > 0:
            model_ext = os.path.splitext(sd_model_path)[1]
            if model_ext == '.safetensors':
                model.load_state_dict(load_file(sd_model_path), strict=False)
            elif model_ext == '.ckpt' or model_ext == '.pth':
                model.load_state_dict(torch.load(sd_model_path)['state_dict'],
                                      strict=False)

        try:
            model.first_stage_model.load_state_dict(torch.load(
                './models/vae-ft-mse-840000-ema-pruned.ckpt')['state_dict'],
                                                    strict=False)
        except Exception:
            print('Warning: We suggest you download the fine-tuned VAE',
                  'otherwise the generation quality will be degraded')

        model.model.diffusion_model.forward = freeu_forward(
            model.model.diffusion_model, *freeu_args)
        self.ddim_v_sampler = DDIMVSampler(model)

    def clear_sd_model(self):
        self.sd_model = None
        self.ddim_v_sampler = None
        torch.cuda.empty_cache()

    def update_detector(self, control_type, canny_low=100, canny_high=200):
        if self.detector_type == control_type:
            return
        if control_type == 'HED':
            self.detector = HEDdetector()
        elif control_type == 'canny':
            canny_detector = CannyDetector()
            low_threshold = canny_low
            high_threshold = canny_high

            def apply_canny(x):
                return canny_detector(x, low_threshold, high_threshold)

            self.detector = apply_canny


global_state = GlobalState()
global_video_path = None
video_frame_count = None


def create_cfg(input_path, prompt, image_resolution, control_strength,
               color_preserve, left_crop, right_crop, top_crop, bottom_crop,
               control_type, low_threshold, high_threshold, ddim_steps, scale,
               seed, sd_model, a_prompt, n_prompt, interval, keyframe_count,
               x0_strength, use_constraints, cross_start, cross_end,
               style_update_freq, warp_start, warp_end, mask_start, mask_end,
               ada_start, ada_end, mask_strength, inner_strength,
               smooth_boundary, loose_cfattn, b1, b2, s1, s2):
    use_warp = 'shape-aware fusion' in use_constraints
    use_mask = 'pixel-aware fusion' in use_constraints
    use_ada = 'color-aware AdaIN' in use_constraints

    if not use_warp:
        warp_start = 1
        warp_end = 0

    if not use_mask:
        mask_start = 1
        mask_end = 0

    if not use_ada:
        ada_start = 1
        ada_end = 0

    input_name = os.path.split(input_path)[-1].split('.')[0]
    frame_count = 2 + keyframe_count * interval
    cfg = RerenderConfig()
    cfg.create_from_parameters(
        input_path,
        os.path.join('result', input_name, 'blend.mp4'),
        prompt,
        a_prompt=a_prompt,
        n_prompt=n_prompt,
        frame_count=frame_count,
        interval=interval,
        crop=[left_crop, right_crop, top_crop, bottom_crop],
        sd_model=sd_model,
        ddim_steps=ddim_steps,
        scale=scale,
        control_type=control_type,
        control_strength=control_strength,
        canny_low=low_threshold,
        canny_high=high_threshold,
        seed=seed,
        image_resolution=image_resolution,
        x0_strength=x0_strength,
        style_update_freq=style_update_freq,
        cross_period=(cross_start, cross_end),
        warp_period=(warp_start, warp_end),
        mask_period=(mask_start, mask_end),
        ada_period=(ada_start, ada_end),
        mask_strength=mask_strength,
        inner_strength=inner_strength,
        smooth_boundary=smooth_boundary,
        color_preserve=color_preserve,
        loose_cfattn=loose_cfattn,
        freeu_args=[b1, b2, s1, s2])
    return cfg


def cfg_to_input(filename):

    cfg = RerenderConfig()
    cfg.create_from_path(filename)
    keyframe_count = (cfg.frame_count - 2) // cfg.interval
    use_constraints = [
        'shape-aware fusion', 'pixel-aware fusion', 'color-aware AdaIN'
    ]

    sd_model = inversed_model_dict.get(cfg.sd_model, 'Stable Diffusion 1.5')

    args = [
        cfg.input_path, cfg.prompt, cfg.image_resolution, cfg.control_strength,
        cfg.color_preserve, *cfg.crop, cfg.control_type, cfg.canny_low,
        cfg.canny_high, cfg.ddim_steps, cfg.scale, cfg.seed, sd_model,
        cfg.a_prompt, cfg.n_prompt, cfg.interval, keyframe_count,
        cfg.x0_strength, use_constraints, *cfg.cross_period,
        cfg.style_update_freq, *cfg.warp_period, *cfg.mask_period,
        *cfg.ada_period, cfg.mask_strength, cfg.inner_strength,
        cfg.smooth_boundary, cfg.loose_cfattn, *cfg.freeu_args
    ]
    return args


def setup_color_correction(image):
    correction_target = cv2.cvtColor(np.asarray(image.copy()),
                                     cv2.COLOR_RGB2LAB)
    return correction_target


def apply_color_correction(correction, original_image):
    image = Image.fromarray(
        cv2.cvtColor(
            exposure.match_histograms(cv2.cvtColor(np.asarray(original_image),
                                                   cv2.COLOR_RGB2LAB),
                                      correction,
                                      channel_axis=2),
            cv2.COLOR_LAB2RGB).astype('uint8'))

    image = blendLayers(image, original_image, BlendType.LUMINOSITY)

    return image


@torch.no_grad()
def process(*args):
    args_wo_process3 = args[:-2]
    first_frame = process1(*args_wo_process3)

    keypath = process2(*args_wo_process3)

    fullpath = process3(*args)

    return first_frame, keypath, fullpath


@torch.no_grad()
def process1(*args):

    global global_video_path
    cfg = create_cfg(global_video_path, *args)
    global global_state
    global_state.update_sd_model(cfg.sd_model, cfg.control_type,
                                 cfg.freeu_args)
    global_state.update_controller(cfg.inner_strength, cfg.mask_period,
                                   cfg.cross_period, cfg.ada_period,
                                   cfg.warp_period, cfg.loose_cfattn)
    global_state.update_detector(cfg.control_type, cfg.canny_low,
                                 cfg.canny_high)
    global_state.processing_state = ProcessingState.FIRST_IMG

    prepare_frames(cfg.input_path, cfg.input_dir, cfg.image_resolution, cfg.crop, cfg.use_limit_device_resolution)

    ddim_v_sampler = global_state.ddim_v_sampler
    model = ddim_v_sampler.model
    detector = global_state.detector
    controller = global_state.controller
    model.control_scales = [cfg.control_strength] * 13

    num_samples = 1
    eta = 0.0
    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]

    with torch.no_grad():
        frame = cv2.imread(imgs[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = HWC3(frame)
        H, W, C = img.shape

        img_ = numpy2tensor(img)

        def generate_first_img(img_, strength):
            encoder_posterior = model.encode_first_stage(img_.cuda())
            x0 = model.get_first_stage_encoding(encoder_posterior).detach()

            detected_map = detector(img)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(
                detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            cond = {
                'c_concat': [control],
                'c_crossattn': [
                    model.get_learned_conditioning(
                        [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
                ]
            }
            un_cond = {
                'c_concat': [control],
                'c_crossattn':
                [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
            }
            shape = (4, H // 8, W // 8)

            controller.set_task('initfirst')
            seed_everything(cfg.seed)

            samples, _ = ddim_v_sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=cfg.scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=strength)
            x_samples = model.decode_first_stage(samples)
            x_samples_np = (
                einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_samples, x_samples_np

        # When not preserve color, draw a different frame at first and use its
        # color to redraw the first frame.
        if not cfg.color_preserve:
            first_strength = -1
        else:
            first_strength = 1 - cfg.x0_strength

        x_samples, x_samples_np = generate_first_img(img_, first_strength)

        if not cfg.color_preserve:
            color_corrections = setup_color_correction(
                Image.fromarray(x_samples_np[0]))
            global_state.color_corrections = color_corrections
            img_ = apply_color_correction(color_corrections,
                                          Image.fromarray(img))
            img_ = to_tensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
            x_samples, x_samples_np = generate_first_img(
                img_, 1 - cfg.x0_strength)

        global_state.first_result = x_samples
        global_state.first_img = img

    Image.fromarray(x_samples_np[0]).save(
        os.path.join(cfg.first_dir, 'first.jpg'))

    return x_samples_np[0]


@torch.no_grad()
def process2(*args):
    global global_state
    global global_video_path

    if global_state.processing_state != ProcessingState.FIRST_IMG:
        raise gr.Error('Please generate the first key image before generating'
                       ' all key images')

    cfg = create_cfg(global_video_path, *args)
    global_state.update_sd_model(cfg.sd_model, cfg.control_type,
                                 cfg.freeu_args)
    global_state.update_detector(cfg.control_type, cfg.canny_low,
                                 cfg.canny_high)
    global_state.processing_state = ProcessingState.KEY_IMGS

    # reset key dir
    shutil.rmtree(cfg.key_dir)
    os.makedirs(cfg.key_dir, exist_ok=True)

    ddim_v_sampler = global_state.ddim_v_sampler
    model = ddim_v_sampler.model
    detector = global_state.detector
    controller = global_state.controller
    flow_model = global_state.flow_model
    model.control_scales = [cfg.control_strength] * 13

    num_samples = 1
    eta = 0.0
    firstx0 = True
    pixelfusion = cfg.use_mask
    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]

    first_result = global_state.first_result
    first_img = global_state.first_img
    pre_result = first_result
    pre_img = first_img

    for i in range(0, min(len(imgs), cfg.frame_count) - 1, cfg.interval):
        cid = i + 1
        print(cid)
        if cid <= (len(imgs) - 1):
            frame = cv2.imread(imgs[cid])
        else:
            frame = cv2.imread(imgs[len(imgs) - 1])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = HWC3(frame)
        H, W, C = img.shape

        if cfg.color_preserve or global_state.color_corrections is None:
            img_ = numpy2tensor(img)
        else:
            img_ = apply_color_correction(global_state.color_corrections,
                                          Image.fromarray(img))
            img_ = to_tensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
        encoder_posterior = model.encode_first_stage(img_.cuda())
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        detected_map = detector(img)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        cond = {
            'c_concat': [control],
            'c_crossattn': [
                model.get_learned_conditioning(
                    [cfg.prompt + ', ' + cfg.a_prompt] * num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [model.get_learned_conditioning([cfg.n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        cond['c_concat'] = [control]
        un_cond['c_concat'] = [control]

        image1 = torch.from_numpy(pre_img).permute(2, 0, 1).float()
        image2 = torch.from_numpy(img).permute(2, 0, 1).float()
        warped_pre, bwd_occ_pre, bwd_flow_pre = get_warped_and_mask(
            flow_model, image1, image2, pre_result, False)
        blend_mask_pre = blur(
            F.max_pool2d(bwd_occ_pre, kernel_size=9, stride=1, padding=4))
        blend_mask_pre = torch.clamp(blend_mask_pre + bwd_occ_pre, 0, 1)

        image1 = torch.from_numpy(first_img).permute(2, 0, 1).float()
        warped_0, bwd_occ_0, bwd_flow_0 = get_warped_and_mask(
            flow_model, image1, image2, first_result, False)
        blend_mask_0 = blur(
            F.max_pool2d(bwd_occ_0, kernel_size=9, stride=1, padding=4))
        blend_mask_0 = torch.clamp(blend_mask_0 + bwd_occ_0, 0, 1)

        if firstx0:
            mask = 1 - F.max_pool2d(blend_mask_0, kernel_size=8)
            controller.set_warp(
                F.interpolate(bwd_flow_0 / 8.0,
                              scale_factor=1. / 8,
                              mode='bilinear'), mask)
        else:
            mask = 1 - F.max_pool2d(blend_mask_pre, kernel_size=8)
            controller.set_warp(
                F.interpolate(bwd_flow_pre / 8.0,
                              scale_factor=1. / 8,
                              mode='bilinear'), mask)

        controller.set_task('keepx0, keepstyle')
        seed_everything(cfg.seed)
        samples, intermediates = ddim_v_sampler.sample(
            cfg.ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=un_cond,
            controller=controller,
            x0=x0,
            strength=1 - cfg.x0_strength)
        direct_result = model.decode_first_stage(samples)

        if not pixelfusion:
            pre_result = direct_result
            pre_img = img
            viz = (
                einops.rearrange(direct_result, 'b c h w -> b h w c') * 127.5 +
                127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        else:

            blend_results = (1 - blend_mask_pre
                             ) * warped_pre + blend_mask_pre * direct_result
            blend_results = (
                1 - blend_mask_0) * warped_0 + blend_mask_0 * blend_results

            bwd_occ = 1 - torch.clamp(1 - bwd_occ_pre + 1 - bwd_occ_0, 0, 1)
            blend_mask = blur(
                F.max_pool2d(bwd_occ, kernel_size=9, stride=1, padding=4))
            blend_mask = 1 - torch.clamp(blend_mask + bwd_occ, 0, 1)

            encoder_posterior = model.encode_first_stage(blend_results)
            xtrg = model.get_first_stage_encoding(
                encoder_posterior).detach()  # * mask
            blend_results_rec = model.decode_first_stage(xtrg)
            encoder_posterior = model.encode_first_stage(blend_results_rec)
            xtrg_rec = model.get_first_stage_encoding(
                encoder_posterior).detach()
            xtrg_ = (xtrg + 1 * (xtrg - xtrg_rec))  # * mask
            blend_results_rec_new = model.decode_first_stage(xtrg_)
            tmp = (abs(blend_results_rec_new - blend_results).mean(
                dim=1, keepdims=True) > 0.25).float()
            mask_x = F.max_pool2d((F.interpolate(
                tmp, scale_factor=1 / 8., mode='bilinear') > 0).float(),
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

            mask = (1 - F.max_pool2d(1 - blend_mask, kernel_size=8)
                    )  # * (1-mask_x)

            if cfg.smooth_boundary:
                noise_rescale = find_flat_region(mask)
            else:
                noise_rescale = torch.ones_like(mask)
            masks = []
            for j in range(cfg.ddim_steps):
                if j <= cfg.ddim_steps * cfg.mask_period[
                        0] or j >= cfg.ddim_steps * cfg.mask_period[1]:
                    masks += [None]
                else:
                    masks += [mask * cfg.mask_strength]

            # mask 3
            # xtrg = ((1-mask_x) *
            #         (xtrg + xtrg - xtrg_rec) + mask_x * samples) * mask
            # mask 2
            # xtrg = (xtrg + 1 * (xtrg - xtrg_rec)) * mask
            xtrg = (xtrg + (1 - mask_x) * (xtrg - xtrg_rec)) * mask  # mask 1

            tasks = 'keepstyle, keepx0'
            if not firstx0:
                tasks += ', updatex0'
            if i % cfg.style_update_freq == 0:
                tasks += ', updatestyle'
            controller.set_task(tasks, 1.0)

            seed_everything(cfg.seed)
            samples, _ = ddim_v_sampler.sample(
                cfg.ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=cfg.scale,
                unconditional_conditioning=un_cond,
                controller=controller,
                x0=x0,
                strength=1 - cfg.x0_strength,
                xtrg=xtrg,
                mask=masks,
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                   127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        Image.fromarray(viz[0]).save(
            os.path.join(cfg.key_dir, f'{cid:04d}.png'))

    key_video_path = os.path.join(cfg.work_dir, 'key.mp4')
    fps = get_fps(cfg.input_path)
    fps //= cfg.interval
    frame_to_video(key_video_path, cfg.key_dir, fps, False)

    return key_video_path


@torch.no_grad()
def process3(*args):
    max_process = args[-2]
    use_poisson = args[-1]
    args = args[:-2]
    global global_video_path
    global global_state
    if global_state.processing_state != ProcessingState.KEY_IMGS:
        raise gr.Error('Please generate key images before propagation')

    global_state.clear_sd_model()

    cfg = create_cfg(global_video_path, *args)

    # reset blend dir
    blend_dir = os.path.join(cfg.work_dir, 'blend')
    if os.path.exists(blend_dir):
        shutil.rmtree(blend_dir)
    os.makedirs(blend_dir, exist_ok=True)

    video_base_dir = cfg.work_dir
    o_video = cfg.output_path
    fps = get_fps(cfg.input_path)

    end_frame = cfg.frame_count - 1
    interval = cfg.interval
    key_dir = os.path.split(cfg.key_dir)[-1]
    o_video_cmd = f'--output {o_video}'
    ps = '-ps' if use_poisson else ''
    cmd = (f'python video_blend.py {video_base_dir} --beg 1 --end {end_frame} '
           f'--itv {interval} --key {key_dir}  {o_video_cmd} --fps {fps} '
           f'--n_proc {max_process} {ps}')
    print(cmd)
    os.system(cmd)

    return o_video


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown('## Rerender A Video')
    with gr.Row():
        with gr.Column():
            input_path = gr.Video(label='Input Video',
                                  source='upload',
                                  format='mp4',
                                  visible=True)
            prompt = gr.Textbox(label='Prompt')
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=2147483647,
                             step=1,
                             value=0,
                             randomize=True)
            run_button = gr.Button(value='Run All')
            with gr.Row():
                run_button1 = gr.Button(value='Run 1st Key Frame')
                run_button2 = gr.Button(value='Run Key Frames')
                run_button3 = gr.Button(value='Run Propagation')
            with gr.Accordion('Advanced options for the 1st frame translation',
                              open=False):
                image_resolution = gr.Slider(label='Frame resolution',
                                             minimum=256,
                                             maximum=768,
                                             value=512,
                                             step=64)
                control_strength = gr.Slider(label='ControlNet strength',
                                             minimum=0.0,
                                             maximum=2.0,
                                             value=1.0,
                                             step=0.01)
                x0_strength = gr.Slider(
                    label='Denoising strength',
                    minimum=0.00,
                    maximum=1.05,
                    value=0.75,
                    step=0.05,
                    info=('0: fully recover the input.'
                          '1.05: fully rerender the input.'))
                color_preserve = gr.Checkbox(
                    label='Preserve color',
                    value=True,
                    info='Keep the color of the input video')
                with gr.Row():
                    left_crop = gr.Slider(label='Left crop length',
                                          minimum=0,
                                          maximum=512,
                                          value=0,
                                          step=1)
                    right_crop = gr.Slider(label='Right crop length',
                                           minimum=0,
                                           maximum=512,
                                           value=0,
                                           step=1)
                with gr.Row():
                    top_crop = gr.Slider(label='Top crop length',
                                         minimum=0,
                                         maximum=512,
                                         value=0,
                                         step=1)
                    bottom_crop = gr.Slider(label='Bottom crop length',
                                            minimum=0,
                                            maximum=512,
                                            value=0,
                                            step=1)
                with gr.Row():
                    control_type = gr.Dropdown(['HED', 'canny'],
                                               label='Control type',
                                               value='HED')
                    low_threshold = gr.Slider(label='Canny low threshold',
                                              minimum=1,
                                              maximum=255,
                                              value=100,
                                              step=1)
                    high_threshold = gr.Slider(label='Canny high threshold',
                                               minimum=1,
                                               maximum=255,
                                               value=200,
                                               step=1)
                ddim_steps = gr.Slider(label='Steps',
                                       minimum=20,
                                       maximum=100,
                                       value=20,
                                       step=20)
                scale = gr.Slider(label='CFG scale',
                                  minimum=0.1,
                                  maximum=30.0,
                                  value=7.5,
                                  step=0.1)
                sd_model_list = list(model_dict.keys())
                sd_model = gr.Dropdown(sd_model_list,
                                       label='Base model',
                                       value='Stable Diffusion 1.5')
                a_prompt = gr.Textbox(label='Added prompt',
                                      value='best quality, extremely detailed')
                n_prompt = gr.Textbox(
                    label='Negative prompt',
                    value=('longbody, lowres, bad anatomy, bad hands, '
                           'missing fingers, extra digit, fewer digits, '
                           'cropped, worst quality, low quality'))
                with gr.Row():
                    b1 = gr.Slider(label='FreeU first-stage backbone factor',
                                   minimum=1,
                                   maximum=1.6,
                                   value=1,
                                   step=0.01,
                                   info='FreeU to enhance texture and color')
                    b2 = gr.Slider(label='FreeU second-stage backbone factor',
                                   minimum=1,
                                   maximum=1.6,
                                   value=1,
                                   step=0.01)
                with gr.Row():
                    s1 = gr.Slider(label='FreeU first-stage skip factor',
                                   minimum=0,
                                   maximum=1,
                                   value=1,
                                   step=0.01)
                    s2 = gr.Slider(label='FreeU second-stage skip factor',
                                   minimum=0,
                                   maximum=1,
                                   value=1,
                                   step=0.01)
            with gr.Accordion('Advanced options for the key fame translation',
                              open=False):
                interval = gr.Slider(
                    label='Key frame frequency (K)',
                    minimum=1,
                    maximum=1,
                    value=1,
                    step=1,
                    info='Uniformly sample the key frames every K frames')
                keyframe_count = gr.Slider(label='Number of key frames',
                                           minimum=1,
                                           maximum=1,
                                           value=1,
                                           step=1)

                use_constraints = gr.CheckboxGroup(
                    [
                        'shape-aware fusion', 'pixel-aware fusion',
                        'color-aware AdaIN'
                    ],
                    label='Select the cross-frame contraints to be used',
                    value=[
                        'shape-aware fusion', 'pixel-aware fusion',
                        'color-aware AdaIN'
                    ]),
                with gr.Row():
                    cross_start = gr.Slider(
                        label='Cross-frame attention start',
                        minimum=0,
                        maximum=1,
                        value=0,
                        step=0.05)
                    cross_end = gr.Slider(label='Cross-frame attention end',
                                          minimum=0,
                                          maximum=1,
                                          value=1,
                                          step=0.05)
                style_update_freq = gr.Slider(
                    label='Cross-frame attention update frequency',
                    minimum=1,
                    maximum=100,
                    value=1,
                    step=1,
                    info=('Update the key and value for '
                          'cross-frame attention every N key frames'))
                loose_cfattn = gr.Checkbox(
                    label='Loose Cross-frame attention',
                    value=True,
                    info='Select to make output better match the input video')
                with gr.Row():
                    warp_start = gr.Slider(label='Shape-aware fusion start',
                                           minimum=0,
                                           maximum=1,
                                           value=0,
                                           step=0.05)
                    warp_end = gr.Slider(label='Shape-aware fusion end',
                                         minimum=0,
                                         maximum=1,
                                         value=0.1,
                                         step=0.05)
                with gr.Row():
                    mask_start = gr.Slider(label='Pixel-aware fusion start',
                                           minimum=0,
                                           maximum=1,
                                           value=0.5,
                                           step=0.05)
                    mask_end = gr.Slider(label='Pixel-aware fusion end',
                                         minimum=0,
                                         maximum=1,
                                         value=0.8,
                                         step=0.05)
                with gr.Row():
                    ada_start = gr.Slider(label='Color-aware AdaIN start',
                                          minimum=0,
                                          maximum=1,
                                          value=0.8,
                                          step=0.05)
                    ada_end = gr.Slider(label='Color-aware AdaIN end',
                                        minimum=0,
                                        maximum=1,
                                        value=1,
                                        step=0.05)
                mask_strength = gr.Slider(label='Pixel-aware fusion strength',
                                          minimum=0,
                                          maximum=1,
                                          value=0.5,
                                          step=0.01)
                inner_strength = gr.Slider(
                    label='Pixel-aware fusion detail level',
                    minimum=0.5,
                    maximum=1,
                    value=0.9,
                    step=0.01,
                    info='Use a low value to prevent artifacts')
                smooth_boundary = gr.Checkbox(
                    label='Smooth fusion boundary',
                    value=True,
                    info='Select to prevent artifacts at boundary')
            with gr.Accordion(
                    'Advanced options for the full video translation',
                    open=False):
                use_poisson = gr.Checkbox(
                    label='Gradient blending',
                    value=True,
                    info=('Blend the output video in gradient, to reduce'
                          ' ghosting artifacts (but may increase flickers)'))
                max_process = gr.Slider(label='Number of parallel processes',
                                        minimum=1,
                                        maximum=16,
                                        value=4,
                                        step=1)

            with gr.Accordion('Example configs', open=True):
                config_dir = 'config'
                config_list = [
                    'real2sculpture.json', 'van_gogh_man.json', 'woman.json'
                ]
                args_list = []
                for config in config_list:
                    try:
                        config_path = os.path.join(config_dir, config)
                        args = cfg_to_input(config_path)
                        args_list.append(args)
                    except FileNotFoundError:
                        # The video file does not exist, skipped
                        pass

                ips = [
                    prompt, image_resolution, control_strength, color_preserve,
                    left_crop, right_crop, top_crop, bottom_crop, control_type,
                    low_threshold, high_threshold, ddim_steps, scale, seed,
                    sd_model, a_prompt, n_prompt, interval, keyframe_count,
                    x0_strength, use_constraints[0], cross_start, cross_end,
                    style_update_freq, warp_start, warp_end, mask_start,
                    mask_end, ada_start, ada_end, mask_strength,
                    inner_strength, smooth_boundary, loose_cfattn, b1, b2, s1,
                    s2
                ]

                gr.Examples(
                    examples=args_list,
                    inputs=[input_path, *ips],
                )

        with gr.Column():
            result_image = gr.Image(label='Output first frame',
                                    type='numpy',
                                    interactive=False)
            result_keyframe = gr.Video(label='Output key frame video',
                                       format='mp4',
                                       interactive=False)
            result_video = gr.Video(label='Output full video',
                                    format='mp4',
                                    interactive=False)

    def input_uploaded(path):
        frame_count = get_frame_count(path)
        if frame_count <= 2:
            raise gr.Error('The input video is too short!'
                           'Please input another video.')

        default_interval = min(10, frame_count - 2)
        max_keyframe = (frame_count - 2) // default_interval

        global video_frame_count
        video_frame_count = frame_count
        global global_video_path
        global_video_path = path

        return gr.Slider.update(value=default_interval,
                                maximum=max_keyframe), gr.Slider.update(
                                    value=max_keyframe, maximum=max_keyframe)

    def input_changed(path):
        frame_count = get_frame_count(path)
        if frame_count <= 2:
            return gr.Slider.update(maximum=1), gr.Slider.update(maximum=1)

        default_interval = min(10, frame_count - 2)
        max_keyframe = (frame_count - 2) // default_interval

        global video_frame_count
        video_frame_count = frame_count
        global global_video_path
        global_video_path = path

        return gr.Slider.update(maximum=max_keyframe), \
            gr.Slider.update(maximum=max_keyframe)

    def interval_changed(interval):
        global video_frame_count
        if video_frame_count is None:
            return gr.Slider.update()

        max_keyframe = (video_frame_count - 2) // interval

        return gr.Slider.update(value=max_keyframe, maximum=max_keyframe)

    input_path.change(input_changed, input_path, [interval, keyframe_count])
    input_path.upload(input_uploaded, input_path, [interval, keyframe_count])
    interval.change(interval_changed, interval, keyframe_count)

    ips_process3 = [*ips, max_process, use_poisson]
    run_button.click(fn=process,
                     inputs=ips_process3,
                     outputs=[result_image, result_keyframe, result_video])
    run_button1.click(fn=process1, inputs=ips, outputs=[result_image])
    run_button2.click(fn=process2, inputs=ips, outputs=[result_keyframe])
    run_button3.click(fn=process3, inputs=ips_process3, outputs=[result_video])

block.launch(server_name='localhost')
