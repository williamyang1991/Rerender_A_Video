import argparse
import os
import random

import cv2
import einops
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
from deps.ControlNet.cldm.cldm import ControlLDM
from deps.ControlNet.cldm.model import create_model, load_state_dict
from deps.gmflow.gmflow.gmflow import GMFlow
from flow.flow_utils import get_warped_and_mask
from src.config import RerenderConfig
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.freeu import freeu_forward
from src.img_util import find_flat_region, numpy2tensor
from src.video_util import frame_to_video, get_fps, prepare_frames

blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
totensor = T.PILToTensor()


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


def rerender(cfg: RerenderConfig, first_img_only: bool, key_video_path: str):

    # Preprocess input
    prepare_frames(cfg.input_path, cfg.input_dir, cfg.image_resolution, cfg.crop, cfg.use_limit_device_resolution)

    # Load models
    if cfg.control_type == 'HED':
        detector = HEDdetector()
    elif cfg.control_type == 'canny':
        canny_detector = CannyDetector()
        low_threshold = cfg.canny_low
        high_threshold = cfg.canny_high

        def apply_canny(x):
            return canny_detector(x, low_threshold, high_threshold)

        detector = apply_canny

    model: ControlLDM = create_model(
        './deps/ControlNet/models/cldm_v15.yaml').cpu()
    if cfg.control_type == 'HED':
        model.load_state_dict(
            load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
    elif cfg.control_type == 'canny':
        model.load_state_dict(
            load_state_dict('./models/control_sd15_canny.pth',
                            location='cuda'))
    model = model.cuda()
    model.control_scales = [cfg.control_strength] * 13

    if cfg.sd_model is not None:
        model_ext = os.path.splitext(cfg.sd_model)[1]
        if model_ext == '.safetensors':
            model.load_state_dict(load_file(cfg.sd_model), strict=False)
        elif model_ext == '.ckpt' or model_ext == '.pth':
            model.load_state_dict(torch.load(cfg.sd_model)['state_dict'],
                                  strict=False)

    try:
        model.first_stage_model.load_state_dict(torch.load(
            './models/vae-ft-mse-840000-ema-pruned.ckpt')['state_dict'],
                                                strict=False)
    except Exception:
        print('Warning: We suggest you download the fine-tuned VAE',
              'otherwise the generation quality will be degraded')

    model.model.diffusion_model.forward = \
        freeu_forward(model.model.diffusion_model, *cfg.freeu_args)
    ddim_v_sampler = DDIMVSampler(model)

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

    num_samples = 1
    ddim_steps = 20
    scale = 7.5

    seed = cfg.seed
    if seed == -1:
        seed = random.randint(0, 65535)
    eta = 0.0

    prompt = cfg.prompt
    a_prompt = cfg.a_prompt
    n_prompt = cfg.n_prompt
    prompt = prompt + ', ' + a_prompt

    style_update_freq = cfg.style_update_freq
    pixelfusion = True
    color_preserve = cfg.color_preserve

    x0_strength = 1 - cfg.x0_strength
    mask_period = cfg.mask_period
    firstx0 = True
    controller = AttentionControl(cfg.inner_strength, cfg.mask_period,
                                  cfg.cross_period, cfg.ada_period,
                                  cfg.warp_period, cfg.loose_cfattn)

    imgs = sorted(os.listdir(cfg.input_dir))
    imgs = [os.path.join(cfg.input_dir, img) for img in imgs]
    if cfg.frame_count >= 0:
        imgs = imgs[:cfg.frame_count]

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
        encoder_posterior = model.encode_first_stage(img_.cuda())
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        detected_map = detector(img)
        detected_map = HWC3(detected_map)
        # For visualization
        detected_img = 255 - detected_map

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        cond = {
            'c_concat': [control],
            'c_crossattn':
            [model.get_learned_conditioning([prompt] * num_samples)]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        controller.set_task('initfirst')
        seed_everything(seed)
        samples, _ = ddim_v_sampler.sample(ddim_steps,
                                           num_samples,
                                           shape,
                                           cond,
                                           verbose=False,
                                           eta=eta,
                                           unconditional_guidance_scale=scale,
                                           unconditional_conditioning=un_cond,
                                           controller=controller,
                                           x0=x0,
                                           strength=x0_strength)
        x_samples = model.decode_first_stage(samples)
        pre_result = x_samples
        pre_img = img
        first_result = pre_result
        first_img = pre_img

        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    color_corrections = setup_color_correction(Image.fromarray(x_samples[0]))
    Image.fromarray(x_samples[0]).save(os.path.join(cfg.first_dir,
                                                    'first.jpg'))
    cv2.imwrite(os.path.join(cfg.first_dir, 'first_edge.jpg'), detected_img)

    if first_img_only:
        exit(0)

    for i in range(0, min(len(imgs), cfg.frame_count) - 1, cfg.interval):
        cid = i + 1
        print(cid)
        if cid <= (len(imgs) - 1):
            frame = cv2.imread(imgs[cid])
        else:
            frame = cv2.imread(imgs[len(imgs) - 1])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = HWC3(frame)

        if color_preserve:
            img_ = numpy2tensor(img)
        else:
            img_ = apply_color_correction(color_corrections,
                                          Image.fromarray(img))
            img_ = totensor(img_).unsqueeze(0)[:, :3] / 127.5 - 1
        encoder_posterior = model.encode_first_stage(img_.cuda())
        x0 = model.get_first_stage_encoding(encoder_posterior).detach()

        detected_map = detector(img)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
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
        seed_everything(seed)
        samples, intermediates = ddim_v_sampler.sample(
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
            strength=x0_strength)
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
            for j in range(ddim_steps):
                if j <= ddim_steps * mask_period[
                        0] or j >= ddim_steps * mask_period[1]:
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
            if i % style_update_freq == 0:
                tasks += ', updatestyle'
            controller.set_task(tasks, 1.0)

            seed_everything(seed)
            samples, _ = ddim_v_sampler.sample(
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
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                   127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        Image.fromarray(viz[0]).save(
            os.path.join(cfg.key_dir, f'{cid:04d}.png'))
    if key_video_path is not None:
        fps = get_fps(cfg.input_path)
        fps //= cfg.interval
        frame_to_video(key_video_path, cfg.key_dir, fps, False)


def postprocess(cfg: RerenderConfig, ne: bool, max_process: int, tmp: bool,
                ps: bool):
    video_base_dir = cfg.work_dir
    o_video = cfg.output_path
    fps = get_fps(cfg.input_path)

    end_frame = cfg.frame_count - 1
    interval = cfg.interval
    key_dir = os.path.split(cfg.key_dir)[-1]
    use_e = '-ne' if ne else ''
    use_tmp = '-tmp' if tmp else ''
    use_ps = '-ps' if ps else ''
    o_video_cmd = f'--output {o_video}'

    cmd = (
        f'python video_blend.py {video_base_dir} --beg 1 --end {end_frame} '
        f'--itv {interval} --key {key_dir} {use_e} {o_video_cmd} --fps {fps} '
        f'--n_proc {max_process} {use_tmp} {use_ps}')
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=None)
    parser.add_argument('--input',
                        type=str,
                        default=None,
                        help='The input path to video.')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--key_video_path', type=str, default=None)
    parser.add_argument('-one',
                        action='store_true',
                        help='Run the first frame with ControlNet only')
    parser.add_argument('-nr',
                        action='store_true',
                        help='Do not run rerender and do postprocessing only')
    parser.add_argument('-nb',
                        action='store_true',
                        help='Do not run postprocessing and run rerender only')
    parser.add_argument(
        '-ne',
        action='store_true',
        help='Do not run ebsynth (use previous ebsynth temporary output)')
    parser.add_argument('-nps',
                        action='store_true',
                        help='Do not run poisson gradient blending')
    parser.add_argument('--n_proc',
                        type=int,
                        default=4,
                        help='The max process count')
    parser.add_argument('--tmp',
                        action='store_true',
                        help='Keep ebsynth temporary output')

    args = parser.parse_args()

    cfg = RerenderConfig()
    if args.cfg is not None:
        cfg.create_from_path(args.cfg)
        if args.input is not None:
            print('Config has been loaded. --input is ignored.')
        if args.output is not None:
            print('Config has been loaded. --output is ignored.')
        if args.prompt is not None:
            print('Config has been loaded. --prompt is ignored.')
    else:
        if args.input is None:
            print('Config not found. --input is required.')
            exit(0)
        if args.output is None:
            print('Config not found. --output is required.')
            exit(0)
        if args.prompt is None:
            print('Config not found. --prompt is required.')
            exit(0)
        cfg.create_from_parameters(args.input, args.output, args.prompt)

    if not args.nr:
        rerender(cfg, args.one, args.key_video_path)
        torch.cuda.empty_cache()
    if not args.nb:
        postprocess(cfg, args.ne, args.n_proc, args.tmp, not args.nps)
