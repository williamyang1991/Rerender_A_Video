import argparse
import json
import os

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

import deps.ControlNet.share  # noqa: F401
import src.path_util  # noqa: F401
from deps.ControlNet.annotator.canny import CannyDetector
from deps.ControlNet.annotator.hed import HEDdetector
from deps.ControlNet.annotator.util import HWC3, resize_image
from deps.ControlNet.cldm.cldm import ControlLDM
from deps.ControlNet.cldm.model import create_model, load_state_dict
from deps.gmflow.gmflow.gmflow import GMFlow
from flow.flow_utils import get_warped_and_mask
from src.controller import AttentionControl
from src.ddim_v_hacked import DDIMVSampler
from src.img_util import find_flat_region, numpy2tensor

# Append deps to path


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


def main(args):
    video_cfg = 'videos.json'
    video_trans_cfg = 'video_trans.json'
    with open(video_cfg, 'r') as fp:
        video_cfg = json.load(fp)
    with open(video_trans_cfg, 'r') as fp:
        video_trans_cfg = json.load(fp)
    video_list = video_cfg['list']
    diffusion_models = video_trans_cfg['models']
    task_config = video_trans_cfg['tasks'][args.tid]

    os.makedirs(task_config['o_tmp_dir'], exist_ok=True)
    os.makedirs(task_config['o_dir'], exist_ok=True)

    blur = T.GaussianBlur(kernel_size=(9, 9), sigma=(18, 18))
    totensor = T.PILToTensor()

    if task_config['control_type'] == 'HED':
        detector = HEDdetector()
    elif task_config['control_type'] == 'canny':
        canny_detector = CannyDetector()
        low_threshold = task_config.get('canny_low', 100)
        high_threshold = task_config.get('canny_high', 200)

        def apply_canny(x):
            return canny_detector(x, low_threshold, high_threshold)

        detector = apply_canny

    model: ControlLDM = create_model('./models/cldm_v15.yaml').cpu()
    if task_config['control_type'] == 'HED':
        model.load_state_dict(
            load_state_dict('./models/control_sd15_hed.pth', location='cuda'))
    elif task_config['control_type'] == 'canny':
        model.load_state_dict(
            load_state_dict('./models/control_sd15_canny.pth',
                            location='cuda'))
    model = model.cuda()

    if task_config['model_id'] >= 0:
        model_cfg = diffusion_models[task_config['model_id']]
        if model_cfg['type'] == 'safetensor':
            model.load_state_dict(load_file(model_cfg['path']), strict=False)
        elif model_cfg['type'] == 'ckpt':
            model.load_state_dict(torch.load(model_cfg['path'])['state_dict'],
                                  strict=False)

    if len(task_config['vae']) > 0:
        model.first_stage_model.load_state_dict(torch.load(
            task_config['vae'])['state_dict'],
                                                strict=False)
    elif task_config['better_vae']:
        model.first_stage_model.load_state_dict(torch.load(
            './models/vae-ft-mse-840000-ema-pruned.ckpt')['state_dict'],
                                                strict=False)
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

    checkpoint = torch.load(
        'deps/gmflow/pretrained/gmflow_sintel-0c07dcb3.pth',
        map_location=lambda storage, loc: storage)
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    flow_model.load_state_dict(weights, strict=False)
    flow_model.eval()

    img_dir = video_list[task_config['v_idx']]['dir'] + '/video'
    frame_count = task_config['frame_count']
    frame_interval = task_config['interval']

    num_samples = 1
    image_resolution = 512
    control_strength = task_config['control_strength']
    # detect_resolution = 512
    ddim_steps = 20
    scale = 7.5
    seed = task_config['seed']
    eta = 0.0
    x0_strength = task_config['x0_strength']

    prompt = task_config['prompt']
    a_prompt = task_config['a_prompt']
    n_prompt = task_config['n_prompt']
    model.control_scales = [control_strength] * 13

    firstx0 = True
    mask_step = [0.5, 0.8]
    style_update_freq = max(10, frame_interval)
    pixelfusion = True
    color_preserve = True

    ada_step = task_config.get('ada_step', 1.8)
    warp_step = task_config.get('warp_step', 0.1)
    controller = AttentionControl(ada_step, warp_step)

    imgs = sorted(os.listdir(img_dir))
    imgs = [os.path.join(img_dir, img) for img in imgs]
    if frame_count >= 0:
        imgs = imgs[:frame_count]

    with torch.no_grad():
        frame = cv2.imread(imgs[0])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(HWC3(frame), image_resolution)
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
        detected = 255 - detected_map

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        cond = {
            'c_concat': [control],
            'c_crossattn': [
                model.get_learned_conditioning([prompt + ', ' + a_prompt] *
                                               num_samples)
            ]
        }
        un_cond = {
            'c_concat': [control],
            'c_crossattn':
            [model.get_learned_conditioning([n_prompt] * num_samples)]
        }
        shape = (4, H // 8, W // 8)

        controller.set_task('initfirst')
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
        x_samples = model.decode_first_stage(samples)
        pre_result = x_samples
        pre_img = img
        first_result = pre_result
        first_img = pre_img

        x_samples = (
            einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
            127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    color_corrections = setup_color_correction(Image.fromarray(x_samples[0]))
    Image.fromarray(x_samples[0]).save(f'{task_config["o_tmp_dir"]}/tmp.jpg')
    cv2.imwrite(f'{task_config["o_tmp_dir"]}/tmp_edge.jpg', detected)

    if args.one:
        exit(0)

    for i in range(frame_count - 1):
        cid = i + 1
        frame = cv2.imread(imgs[i + 1])
        if i % frame_interval != 0:
            continue
        print(i + 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = resize_image(HWC3(frame), image_resolution)

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
        # For visualization
        if args.t:
            detected = 255 - detected_map
            cv2.imwrite(f'{task_config["o_tmp_dir"]}/tmp_edge_{cid:04d}.jpg',
                        detected)

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
            noise_rescale = find_flat_region(mask)
            masks = []
            for i in range(ddim_steps):
                if i <= ddim_steps * mask_step[
                        0] or i >= ddim_steps * mask_step[1]:
                    masks += [None]
                else:
                    masks += [mask * 0.5]

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
                strength=x0_strength,
                xtrg=xtrg,
                mask=masks,
                noise_rescale=noise_rescale)
            x_samples = model.decode_first_stage(samples)
            pre_result = x_samples
            pre_img = img

            viz = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 +
                   127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        Image.fromarray(viz[0]).save(f'{task_config["o_dir"]}/{cid:04d}.jpg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tid', type=int)
    parser.add_argument('-t', action='store_true')
    parser.add_argument('-one', action='store_true')
    args = parser.parse_args()
    main(args)
