import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gmflow_dir = os.path.join(parent_dir, 'deps/gmflow')
sys.path.insert(0, gmflow_dir)

from gmflow.gmflow import GMFlow  # noqa: E702 E402 F401
from utils.utils import InputPadder  # noqa: E702 E402


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def bilinear_sample(img,
                    sample_coords,
                    mode='bilinear',
                    padding_mode='zeros',
                    return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img,
                        grid,
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (
            y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def flow_warp(feature,
              flow,
              mask=False,
              mode='bilinear',
              padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w).to(flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature,
                           grid,
                           mode=mode,
                           padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow,
                                       bwd_flow,
                                       alpha=0.01,
                                       beta=0.5):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow
    # (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow,
                                                        dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ, bwd_occ


@torch.no_grad()
def get_warped_and_mask(flow_model,
                        image1,
                        image2,
                        image3=None,
                        pixel_consistency=False):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
    results_dict = flow_model(image1,
                              image2,
                              attn_splits_list=[2],
                              corr_radius_list=[-1],
                              prop_radius_list=[-1],
                              pred_bidir_flow=True)
    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
    fwd_occ, bwd_occ = forward_backward_consistency_check(
        fwd_flow, bwd_flow)  # [1, H, W] float
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(
            bwd_occ +
            (abs(image2 - warped_image1).mean(dim=1) > 255 * 0.25).float(), 0,
            1).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow


class FlowCalc():

    def __init__(self, model_path='./models/gmflow_sintel-0c07dcb3.pth'):
        flow_model = GMFlow(
            feature_channels=128,
            num_scales=1,
            upsample_factor=8,
            num_head=1,
            attention_type='swin',
            ffn_dim_expansion=4,
            num_transformer_layers=6,
        ).to('cuda')

        checkpoint = torch.load(model_path,
                                map_location=lambda storage, loc: storage)
        weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
        flow_model.load_state_dict(weights, strict=False)
        flow_model.eval()
        self.model = flow_model

    @torch.no_grad()
    def get_flow(self, image1, image2, save_path=None):

        if save_path is not None and os.path.exists(save_path):
            bwd_flow = read_flow(save_path)
            return bwd_flow

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        padder = InputPadder(image1.shape, padding_factor=8)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        results_dict = self.model(image1,
                                  image2,
                                  attn_splits_list=[2],
                                  corr_radius_list=[-1],
                                  prop_radius_list=[-1],
                                  pred_bidir_flow=True)
        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
        fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
        bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
        fwd_occ, bwd_occ = forward_backward_consistency_check(
            fwd_flow, bwd_flow)  # [1, H, W] float
        if save_path is not None:
            flow_np = bwd_flow.cpu().numpy()
            np.save(save_path, flow_np)
            mask_path = os.path.splitext(save_path)[0] + '.png'
            bwd_occ = bwd_occ.cpu().permute(1, 2, 0).to(
                torch.long).numpy() * 255
            cv2.imwrite(mask_path, bwd_occ)

        return bwd_flow

    @torch.no_grad()
    def get_mask(self, image1, image2, save_path=None):

        if save_path is not None:
            mask_path = os.path.splitext(save_path)[0] + '.png'
            if os.path.exists(mask_path):
                return read_mask(mask_path)

        image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
        image2 = torch.from_numpy(image2).permute(2, 0, 1).float()
        padder = InputPadder(image1.shape, padding_factor=8)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        results_dict = self.model(image1,
                                  image2,
                                  attn_splits_list=[2],
                                  corr_radius_list=[-1],
                                  prop_radius_list=[-1],
                                  pred_bidir_flow=True)
        flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]
        fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)  # [1, 2, H, W]
        bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)  # [1, 2, H, W]
        fwd_occ, bwd_occ = forward_backward_consistency_check(
            fwd_flow, bwd_flow)  # [1, H, W] float
        if save_path is not None:
            flow_np = bwd_flow.cpu().numpy()
            np.save(save_path, flow_np)
            mask_path = os.path.splitext(save_path)[0] + '.png'
            bwd_occ = bwd_occ.cpu().permute(1, 2, 0).to(
                torch.long).numpy() * 255
            cv2.imwrite(mask_path, bwd_occ)

        return bwd_occ

    def warp(self, img, flow, mode='bilinear'):
        expand = False
        if len(img.shape) == 2:
            expand = True
            img = np.expand_dims(img, 2)

        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        dtype = img.dtype
        img = img.to(torch.float)
        res = flow_warp(img, flow, mode=mode)
        res = res.to(dtype)
        res = res[0].cpu().permute(1, 2, 0).numpy()
        if expand:
            res = res[:, :, 0]
        return res


def read_flow(save_path):
    flow_np = np.load(save_path)
    bwd_flow = torch.from_numpy(flow_np)
    return bwd_flow


def read_mask(save_path):
    mask_path = os.path.splitext(save_path)[0] + '.png'
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


flow_calc = FlowCalc()
